#!/usr/bin/env python3
"""
Flood Impact-Based Forecasting using Bayesian Networks - Version 1

This module implements a Bayesian Network for short-term flood risk assessment
combining:
1. IMERG observations (last 7 days) - antecedent rainfall conditions
2. GEFS ensemble forecasts (14 days) - probabilistic precipitation forecast
3. [Future] ECMWF ensemble forecasts - multi-model approach

The system uses arbitrary precipitation thresholds for empirical probability
calculation (pixel-wise thresholds can be added in future versions).

Data Sources:
- IMERG: NASA GES DISC via earthaccess (GPM_3IMERGDE)
- GEFS: AWS S3 via grib-index-kerchunk streaming

Author: ICPAC IBF Team
Date: January 2026
"""

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Geospatial imports
try:
    import xarray as xr
    import geopandas as gpd
    import regionmask
    HAS_GEO_DEPS = True
except ImportError:
    HAS_GEO_DEPS = False

try:
    import xesmf as xe
    HAS_XESMF = True
except ImportError:
    HAS_XESMF = False


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Country code mapping for East Africa
COUNTRY_CODE_MAP = {
    'BDI': 'Burundi', 'DJI': 'Djibouti', 'ERI': 'Eritrea',
    'ETH': 'Ethiopia', 'KEN': 'Kenya', 'RWA': 'Rwanda',
    'SOM': 'Somalia', 'SSD': 'South Sudan', 'SDN': 'Sudan',
    'TZA': 'Tanzania', 'UGA': 'Uganda'
}

# Precipitation thresholds for 24-hour accumulated rainfall (mm)
# These are arbitrary thresholds - can be replaced with return period thresholds later
PRECIP_THRESHOLDS_24H = {
    'light': 5,      # Light rainfall
    'moderate': 25,  # Moderate rainfall
    'heavy': 50,     # Heavy rainfall
    'very_heavy': 75,  # Very heavy rainfall
    'extreme': 100,  # Extreme rainfall
    'exceptional': 125  # Exceptional rainfall
}

# Return period-style threshold names for BN compatibility
FLOOD_RETURN_PERIODS = ['light', 'moderate', 'heavy', 'very_heavy', 'extreme']

# Antecedent rainfall thresholds (7-day accumulated, mm)
ANTECEDENT_THRESHOLDS = {
    'dry': 10,
    'normal': 30,
    'wet': 60,
    'very_wet': 100,
    'saturated': float('inf')
}

# Observation window configuration
DEFAULT_OBS_DAYS = 7
OBS_TEMPORAL_WEIGHTS = [0.25, 0.20, 0.15, 0.13, 0.10, 0.09, 0.08]  # Day -1 to Day -7

# Forecast accumulation periods (days)
FORECAST_ACCUM_PERIODS = [3, 7, 14]

# East Africa domain
EA_BOUNDS = {
    'lat_min': -12,
    'lat_max': 23,
    'lon_min': 21,
    'lon_max': 53
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def categorize_antecedent_rainfall(rainfall_mm: float) -> str:
    """Categorize 7-day accumulated rainfall into antecedent condition."""
    if np.isnan(rainfall_mm):
        return 'Normal'
    if rainfall_mm < ANTECEDENT_THRESHOLDS['dry']:
        return 'Dry'
    elif rainfall_mm < ANTECEDENT_THRESHOLDS['normal']:
        return 'Normal'
    elif rainfall_mm < ANTECEDENT_THRESHOLDS['wet']:
        return 'Wet'
    elif rainfall_mm < ANTECEDENT_THRESHOLDS['very_wet']:
        return 'Very_Wet'
    else:
        return 'Saturated'


def compute_rainfall_trend(daily_values: List[float]) -> Tuple[str, float]:
    """
    Compute rainfall trend from daily values (most recent first).

    Returns:
        Tuple of (trend_category, trend_magnitude)
    """
    if len(daily_values) < 2:
        return 'Stable', 0.0

    values = [v for v in daily_values if not np.isnan(v)]
    if len(values) < 2:
        return 'Stable', 0.0

    # Reverse so oldest is first for regression
    values = list(reversed(values))
    x = np.arange(len(values))

    slope = np.polyfit(x, values, 1)[0]

    # Normalize slope by mean rainfall
    mean_val = np.mean(values) if np.mean(values) > 0 else 1
    normalized_slope = slope / mean_val

    if normalized_slope < -0.15:
        return 'Decreasing', float(normalized_slope)
    elif normalized_slope > 0.15:
        return 'Increasing', float(normalized_slope)
    return 'Stable', float(normalized_slope)


def compute_forecast_agreement(eprob1: float, eprob2: float) -> str:
    """
    Compute agreement level between two forecast sources.

    Args:
        eprob1: Exceedance probability from first model (0-1)
        eprob2: Exceedance probability from second model (0-1)

    Returns:
        Agreement category: 'Low', 'Medium', 'High'
    """
    diff = abs(eprob1 - eprob2)
    if diff <= 0.15:
        return 'High'
    elif diff <= 0.30:
        return 'Medium'
    else:
        return 'Low'


def compute_empirical_probability(
    ensemble_precip: np.ndarray,
    threshold: float
) -> float:
    """
    Compute empirical exceedance probability from ensemble.

    Args:
        ensemble_precip: Array of precipitation values from ensemble members
        threshold: Precipitation threshold (mm)

    Returns:
        Probability of exceeding threshold (0-1)
    """
    valid_members = ensemble_precip[~np.isnan(ensemble_precip)]
    if len(valid_members) == 0:
        return 0.0

    n_exceeding = np.sum(valid_members >= threshold)
    return n_exceeding / len(valid_members)


def compute_severity_index(eprobs: Dict[str, float]) -> float:
    """
    Compute flood severity index from exceedance probabilities.

    Uses weighted combination of different threshold exceedances.
    """
    # Base on moderate threshold probability
    base_prob = eprobs.get('eprob_moderate', eprobs.get('eprob_heavy', 0.0))

    boost = 0.0
    if eprobs.get('eprob_heavy', 0) > 0.3:
        boost += 0.05
    if eprobs.get('eprob_very_heavy', 0) > 0.2:
        boost += 0.05
    if eprobs.get('eprob_extreme', 0) > 0.1:
        boost += 0.10

    return min(1.0, base_prob + boost)


def posterior_confidence(probs) -> float:
    """
    Sharpness of a posterior: 1 - H/H_max in [0, 1] (H_max = log k for k states).

    This replaces ``confidence = np.max(action_probs)``, which was read off the
    action node -- a node the Julia port marks deprecated (Layer-2 leakage), so
    the emitted ``confidence`` reported how peaked a deprecated distribution was
    rather than how uncertain the hazard belief is.  A boundary with a flat,
    genuinely uncertain risk posterior could report high confidence.

    Entropy is the honest measure: a flat posterior (we know nothing) scores 0
    regardless of which state happens to edge ahead, and only a genuinely peaked
    belief scores near 1.  It is a statement about the *quality of the belief*.

    Computed on ``risk_probs`` so that this matches `posterior_confidence` in
    `flood_bn_ibf_v1.jl` exactly -- see `python_julia_bn_comparison.md` section
    2.6.  The two implementations must not drift again.
    """
    p = np.asarray(probs, dtype=float).ravel()
    k = p.size
    if k <= 1:
        return 1.0
    total = p.sum()
    if not total > 0:
        return 0.0
    p = p / total
    H = -np.sum(p * np.log(np.maximum(p, 1e-12)))
    return float(np.clip(1.0 - H / np.log(k), 0.0, 1.0))


# ============================================================================
# DATA LOADER
# ============================================================================

class FloodDataLoaderV1:
    """
    Data loader for flood risk assessment.

    Loads and processes:
    - IMERG observations (antecedent rainfall)
    - GEFS ensemble forecasts (probabilistic precipitation)
    """

    def __init__(
        self,
        boundaries_path: str,
        imerg_data_dir: str = "./imerg_data",
        gefs_data_dir: str = None,
        n_obs_days: int = DEFAULT_OBS_DAYS
    ):
        if not HAS_GEO_DEPS:
            raise ImportError("Geospatial dependencies not installed")

        self.boundaries_path = Path(boundaries_path)
        self.imerg_data_dir = Path(imerg_data_dir)
        self.gefs_data_dir = Path(gefs_data_dir) if gefs_data_dir else None
        self.n_obs_days = n_obs_days

        # Load boundaries
        self.boundaries = self._load_boundaries()
        self.n_boundaries = len(self.boundaries)

        logger.info(f"FloodDataLoaderV1 initialized")
        logger.info(f"  Boundaries: {self.n_boundaries}")
        logger.info(f"  IMERG dir: {self.imerg_data_dir}")

    def _load_boundaries(self) -> gpd.GeoDataFrame:
        """Load and prepare admin boundaries."""
        gdf = gpd.read_file(self.boundaries_path)

        if 'GID_1' in gdf.columns:
            gdf['id'] = gdf['GID_1']
            gdf['country_code'] = gdf['GID_1'].str.split('.').str[0]
            gdf['country'] = gdf['country_code'].map(COUNTRY_CODE_MAP).fillna('Unknown')
        else:
            gdf['id'] = [f'ADMIN_{i:03d}' for i in range(len(gdf))]
            gdf['country'] = 'Unknown'

        if 'NAME_1' in gdf.columns:
            gdf['name'] = gdf['NAME_1']
        else:
            gdf['name'] = gdf['id']

        gdf['centroid_lon'] = gdf.geometry.centroid.x
        gdf['centroid_lat'] = gdf.geometry.centroid.y

        logger.info(f"Loaded {len(gdf)} boundaries from {self.boundaries_path.name}")
        return gdf

    def load_imerg_observations(
        self,
        target_date: datetime,
        n_days: int = None
    ) -> Tuple[xr.Dataset, List[datetime]]:
        """
        Load IMERG daily precipitation for observation window.

        Args:
            target_date: End date of observation window
            n_days: Number of days to load (default: self.n_obs_days)

        Returns:
            Tuple of (combined dataset, list of available dates)
        """
        if n_days is None:
            n_days = self.n_obs_days

        datasets = []
        available_dates = []

        for i in range(n_days):
            date = target_date - timedelta(days=i+1)
            date_str = date.strftime('%Y%m%d')

            # Look for IMERG file
            pattern = f"*{date_str}*.nc4"
            files = list(self.imerg_data_dir.glob(pattern))

            if files:
                try:
                    ds = xr.open_dataset(files[0])
                    datasets.append(ds)
                    available_dates.append(date)
                    logger.debug(f"Loaded IMERG for {date_str}")
                except Exception as e:
                    logger.warning(f"Failed to load IMERG {date_str}: {e}")

        if not datasets:
            logger.warning("No IMERG data found")
            return None, []

        # Combine datasets
        combined = xr.concat(datasets, dim='time')
        logger.info(f"Loaded {len(available_dates)}/{n_days} IMERG days")

        return combined, available_dates

    def compute_antecedent_stats(
        self,
        imerg_ds: xr.Dataset,
        available_dates: List[datetime]
    ) -> xr.Dataset:
        """
        Compute antecedent rainfall statistics from IMERG data.

        Returns dataset with:
        - accumulated_7d: 7-day accumulated precipitation
        - daily_values: Daily precipitation values
        - trend: Rainfall trend
        """
        if imerg_ds is None:
            return None

        # Get precipitation variable
        if 'precipitation' in imerg_ds.data_vars:
            precip = imerg_ds['precipitation']
        elif 'precipitationCal' in imerg_ds.data_vars:
            precip = imerg_ds['precipitationCal']
        else:
            var_name = list(imerg_ds.data_vars)[0]
            precip = imerg_ds[var_name]
            logger.warning(f"Using {var_name} as precipitation variable")

        # Compute 7-day accumulation
        accum_7d = precip.sum(dim='time')

        # Create result dataset
        result = xr.Dataset({
            'accumulated_7d': accum_7d,
        })

        result.attrs['n_days'] = len(available_dates)
        result.attrs['start_date'] = min(available_dates).strftime('%Y-%m-%d')
        result.attrs['end_date'] = max(available_dates).strftime('%Y-%m-%d')

        return result

    def load_gefs_probabilities(
        self,
        target_date: datetime,
        run_hour: str = '00',
        accum_days: int = 7
    ) -> Optional[xr.Dataset]:
        """
        Load pre-computed GEFS exceedance probabilities.

        Note: This expects probabilities pre-computed from GEFS ensemble
        using the run_gefs_data_streaming_v2.py script.

        Args:
            target_date: Forecast initialization date
            run_hour: Model run hour ('00', '06', '12', '18')
            accum_days: Accumulation period in days

        Returns:
            Dataset with exceedance probabilities for each threshold
        """
        if self.gefs_data_dir is None:
            logger.warning("GEFS data directory not specified")
            return None

        date_str = target_date.strftime('%Y%m%d')

        # Look for zarr or netcdf files with probabilities
        patterns = [
            f"gefs_eprob_{date_str}_{run_hour}z.zarr",
            f"gefs_eprob_{date_str}_{run_hour}z.nc",
            f"gefs_probability_{date_str}_{run_hour}z.nc"
        ]

        for pattern in patterns:
            filepath = self.gefs_data_dir / pattern
            if filepath.exists():
                try:
                    if filepath.suffix == '.zarr':
                        ds = xr.open_zarr(filepath)
                    else:
                        ds = xr.open_dataset(filepath)
                    logger.info(f"Loaded GEFS probabilities from {filepath.name}")
                    return ds
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")

        logger.warning(f"No GEFS probability files found for {date_str}")
        return None

    def extract_boundary_data(
        self,
        data: xr.DataArray,
        method: str = 'mean'
    ) -> pd.DataFrame:
        """
        Extract values for each boundary from gridded data.

        Args:
            data: Gridded data array (lat, lon)
            method: Aggregation method ('mean', 'max', 'sum')

        Returns:
            DataFrame with boundary-level values
        """
        # Normalize coordinate names
        if 'latitude' in data.coords:
            data = data.rename({'latitude': 'lat', 'longitude': 'lon'})

        # Create region mask
        regions = regionmask.from_geopandas(self.boundaries, names='id', abbrevs='id')

        # Handle coordinate order
        if 'lat' in data.dims and 'lon' in data.dims:
            mask = regions.mask(data.lon, data.lat)
        else:
            logger.warning("Cannot create mask - missing lat/lon dimensions")
            return None

        results = []

        for idx, row in self.boundaries.iterrows():
            boundary_id = row['id']

            try:
                region_num = regions.map_keys(boundary_id)
                boundary_mask = mask.values == region_num
                n_pixels = np.sum(boundary_mask)

                if n_pixels > 0:
                    masked_values = data.values[boundary_mask]

                    if method == 'mean':
                        value = float(np.nanmean(masked_values))
                    elif method == 'max':
                        value = float(np.nanmax(masked_values))
                    elif method == 'sum':
                        value = float(np.nansum(masked_values))
                    else:
                        value = float(np.nanmean(masked_values))

                    extraction_method = 'area'
                else:
                    # Centroid-based extraction
                    value = float(data.sel(
                        lat=row['centroid_lat'],
                        lon=row['centroid_lon'],
                        method='nearest'
                    ).values)
                    extraction_method = 'centroid'
                    n_pixels = 0

                results.append({
                    'boundary_id': boundary_id,
                    'boundary_name': row['name'],
                    'country': row['country'],
                    'value': value,
                    'n_pixels': n_pixels,
                    'extraction_method': extraction_method
                })

            except Exception as e:
                logger.warning(f"Failed to extract for {boundary_id}: {e}")
                results.append({
                    'boundary_id': boundary_id,
                    'boundary_name': row['name'],
                    'country': row['country'],
                    'value': np.nan,
                    'n_pixels': 0,
                    'extraction_method': 'failed'
                })

        return pd.DataFrame(results)

    def prepare_analysis_data(
        self,
        target_date: datetime,
        gefs_run_hour: str = '00',
        forecast_accum_days: int = 7
    ) -> Tuple[List[Dict], Dict]:
        """
        Prepare all data for flood risk analysis.

        Args:
            target_date: Analysis date
            gefs_run_hour: GEFS model run hour
            forecast_accum_days: Forecast accumulation period

        Returns:
            Tuple of (boundary_data_list, metadata_dict)
        """
        logger.info("="*60)
        logger.info(f"PREPARING FLOOD ANALYSIS DATA")
        logger.info(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
        logger.info("="*60)

        # Load IMERG observations
        logger.info(f"\nStep 1: Loading IMERG observations ({self.n_obs_days} days)...")
        imerg_ds, available_dates = self.load_imerg_observations(target_date)

        if imerg_ds is not None:
            antecedent_stats = self.compute_antecedent_stats(imerg_ds, available_dates)
            logger.info(f"  Loaded {len(available_dates)} days of IMERG data")
        else:
            antecedent_stats = None
            logger.warning("  No IMERG data available")

        # Load GEFS probabilities (if available)
        logger.info(f"\nStep 2: Loading GEFS probabilities...")
        gefs_probs = self.load_gefs_probabilities(
            target_date, gefs_run_hour, forecast_accum_days
        )

        # Extract boundary-level data
        logger.info(f"\nStep 3: Extracting boundary-level data...")

        boundaries_data = []

        for idx, row in self.boundaries.iterrows():
            boundary_id = row['id']
            boundary_name = row['name']
            country = row['country']
            centroid_lat = row['centroid_lat']
            centroid_lon = row['centroid_lon']

            boundary_data = {
                'id': boundary_id,
                'name': boundary_name,
                'country': country,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
            }

            # Add antecedent rainfall data
            if antecedent_stats is not None:
                try:
                    accum_7d = float(antecedent_stats['accumulated_7d'].sel(
                        lat=centroid_lat, lon=centroid_lon, method='nearest'
                    ).values)
                except:
                    accum_7d = np.nan

                boundary_data['antecedent_rainfall_mm'] = accum_7d
                boundary_data['antecedent_category'] = categorize_antecedent_rainfall(accum_7d)
            else:
                boundary_data['antecedent_rainfall_mm'] = np.nan
                boundary_data['antecedent_category'] = 'Unknown'

            # Add GEFS probabilities (placeholder for now)
            # In real implementation, extract from gefs_probs dataset
            boundary_data['gefs_eprob_moderate'] = np.nan
            boundary_data['ecmwf_eprob_heavy'] = np.nan
            boundary_data['gefs_eprob_very_heavy'] = np.nan
            boundary_data['gefs_eprob_extreme'] = np.nan

            # Placeholder for ECMWF (future)
            boundary_data['ecmwf_eprob_moderate'] = np.nan
            boundary_data['ecmwf_eprob_heavy'] = np.nan

            # Compute derived metrics
            boundary_data['rainfall_trend'] = 'Stable'
            boundary_data['rainfall_trend_magnitude'] = 0.0
            boundary_data['spatial_coverage'] = 0.0
            boundary_data['forecast_agreement'] = 'Medium'

            boundaries_data.append(boundary_data)

            if (idx + 1) % 50 == 0:
                logger.info(f"  Processed {idx + 1}/{self.n_boundaries}...")

        metadata = {
            'target_date': target_date.strftime('%Y-%m-%d'),
            'obs_days': self.n_obs_days,
            'obs_days_available': len(available_dates) if available_dates else 0,
            'forecast_accum_days': forecast_accum_days,
            'gefs_run_hour': gefs_run_hour,
            'n_boundaries': len(boundaries_data),
            'thresholds': PRECIP_THRESHOLDS_24H
        }

        logger.info(f"\nPrepared data for {len(boundaries_data)} boundaries")

        return boundaries_data, metadata


# ============================================================================
# BAYESIAN NETWORK
# ============================================================================

class FloodBayesianNetworkV1:
    """
    Bayesian Network for flood risk assessment.

    Network structure:
    - antecedent_rainfall → risk_level
    - gefs_exceedance_prob → risk_level
    - spatial_coverage → risk_level
    - rainfall_trend → risk_level
    - forecast_agreement → risk_level (confidence adjustment)
    - risk_level → action
    """

    def __init__(self, include_agreement_node: bool = True):
        self.include_agreement_node = include_agreement_node

        edges = [
            ('antecedent_rainfall', 'risk_level'),
            ('exceedance_prob', 'risk_level'),
            ('spatial_coverage', 'risk_level'),
            ('rainfall_trend', 'risk_level'),
            ('risk_level', 'action')
        ]

        if include_agreement_node:
            edges.append(('forecast_agreement', 'risk_level'))

        self.model = BayesianNetwork(edges)
        self._setup_cpds()

        logger.info("FloodBayesianNetworkV1 initialized")

    def _setup_cpds(self):
        """Setup Conditional Probability Distributions."""

        # Antecedent rainfall CPD (prior)
        antecedent_cpd = TabularCPD(
            'antecedent_rainfall', 5,
            [[0.20], [0.30], [0.25], [0.15], [0.10]],
            state_names={'antecedent_rainfall': ['Dry', 'Normal', 'Wet', 'Very_Wet', 'Saturated']}
        )

        # Exceedance probability CPD (prior)
        exceed_cpd = TabularCPD(
            'exceedance_prob', 5,
            [[0.30], [0.25], [0.20], [0.15], [0.10]],
            state_names={'exceedance_prob': ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']}
        )

        # Spatial coverage CPD (prior)
        spatial_cpd = TabularCPD(
            'spatial_coverage', 3,
            [[0.40], [0.35], [0.25]],
            state_names={'spatial_coverage': ['Localized', 'Moderate', 'Widespread']}
        )

        # Rainfall trend CPD (prior)
        trend_cpd = TabularCPD(
            'rainfall_trend', 3,
            [[0.30], [0.45], [0.25]],
            state_names={'rainfall_trend': ['Decreasing', 'Stable', 'Increasing']}
        )

        cpds = [antecedent_cpd, exceed_cpd, spatial_cpd, trend_cpd]

        if self.include_agreement_node:
            agreement_cpd = TabularCPD(
                'forecast_agreement', 3,
                [[0.20], [0.50], [0.30]],
                state_names={'forecast_agreement': ['Low', 'Medium', 'High']}
            )
            cpds.append(agreement_cpd)

        # Risk level CPD
        risk_cpd = self._create_risk_cpt()
        cpds.append(risk_cpd)

        # Action CPD
        action_cpd = TabularCPD(
            'action', 4,
            np.array([
                [0.95, 0.15, 0.00, 0.00, 0.00],  # Monitor
                [0.05, 0.80, 0.20, 0.05, 0.00],  # Alert
                [0.00, 0.05, 0.75, 0.25, 0.05],  # Prepare
                [0.00, 0.00, 0.05, 0.70, 0.95],  # Act
            ]),
            evidence=['risk_level'],
            evidence_card=[5],
            state_names={
                'action': ['Monitor', 'Alert', 'Prepare', 'Act'],
                'risk_level': ['Minimal', 'Low', 'Moderate', 'High', 'Extreme']
            }
        )
        cpds.append(action_cpd)

        for cpd in cpds:
            self.model.add_cpds(cpd)

        self.model.check_model()
        logger.info("CPDs configured and validated")

    def _create_risk_cpt(self) -> TabularCPD:
        """Create risk level CPT based on expert rules."""

        if self.include_agreement_node:
            n_combinations = 5 * 5 * 3 * 3 * 3  # 675
            evidence = ['antecedent_rainfall', 'exceedance_prob', 'spatial_coverage',
                       'rainfall_trend', 'forecast_agreement']
            evidence_card = [5, 5, 3, 3, 3]
            state_names = {
                'risk_level': ['Minimal', 'Low', 'Moderate', 'High', 'Extreme'],
                'antecedent_rainfall': ['Dry', 'Normal', 'Wet', 'Very_Wet', 'Saturated'],
                'exceedance_prob': ['Very_Low', 'Low', 'Medium', 'High', 'Very_High'],
                'spatial_coverage': ['Localized', 'Moderate', 'Widespread'],
                'rainfall_trend': ['Decreasing', 'Stable', 'Increasing'],
                'forecast_agreement': ['Low', 'Medium', 'High']
            }
        else:
            n_combinations = 5 * 5 * 3 * 3  # 225
            evidence = ['antecedent_rainfall', 'exceedance_prob', 'spatial_coverage', 'rainfall_trend']
            evidence_card = [5, 5, 3, 3]
            state_names = {
                'risk_level': ['Minimal', 'Low', 'Moderate', 'High', 'Extreme'],
                'antecedent_rainfall': ['Dry', 'Normal', 'Wet', 'Very_Wet', 'Saturated'],
                'exceedance_prob': ['Very_Low', 'Low', 'Medium', 'High', 'Very_High'],
                'spatial_coverage': ['Localized', 'Moderate', 'Widespread'],
                'rainfall_trend': ['Decreasing', 'Stable', 'Increasing']
            }

        cpt = np.zeros((5, n_combinations))
        idx = 0

        if self.include_agreement_node:
            for agreement in range(3):
                for trend in range(3):
                    for spatial in range(3):
                        for exceed in range(5):
                            for antecedent in range(5):
                                cpt[:, idx] = self._compute_risk_probs(
                                    antecedent, exceed, spatial, trend, agreement
                                )
                                idx += 1
        else:
            for trend in range(3):
                for spatial in range(3):
                    for exceed in range(5):
                        for antecedent in range(5):
                            cpt[:, idx] = self._compute_risk_probs(
                                antecedent, exceed, spatial, trend, 2  # High agreement default
                            )
                            idx += 1

        return TabularCPD('risk_level', 5, cpt, evidence=evidence,
                         evidence_card=evidence_card, state_names=state_names)

    def _compute_risk_probs(
        self,
        antecedent: int,
        exceed: int,
        spatial: int,
        trend: int,
        agreement: int = 2
    ) -> np.ndarray:
        """
        Compute flood risk probabilities based on expert rules.

        Args:
            antecedent: 0=Dry, 1=Normal, 2=Wet, 3=Very_Wet, 4=Saturated
            exceed: 0=Very_Low, 1=Low, 2=Medium, 3=High, 4=Very_High
            spatial: 0=Localized, 1=Moderate, 2=Widespread
            trend: 0=Decreasing, 1=Stable, 2=Increasing
            agreement: 0=Low, 1=Medium, 2=High
        """
        # Base risk score (antecedent has lower weight than forecast for floods)
        base_risk = (antecedent * 0.30 + exceed * 0.55)

        # Spatial modifier
        if spatial == 2:  # Widespread
            base_risk += 0.5
        elif spatial == 1:  # Moderate
            base_risk += 0.25

        # Trend modifier (increasing rainfall = higher risk)
        if trend == 2:  # Increasing
            base_risk += 0.35
        elif trend == 0:  # Decreasing
            base_risk -= 0.30

        # Expert rules for extreme scenarios

        # Rule 1: Saturated + High/Very_High exceedance + Increasing
        if antecedent == 4 and exceed >= 3 and trend == 2:
            if spatial >= 1:
                probs = np.array([0.0, 0.0, 0.05, 0.20, 0.75])
            else:
                probs = np.array([0.0, 0.0, 0.10, 0.40, 0.50])

        # Rule 2: Very_Wet/Saturated + High exceedance + Increasing
        elif antecedent >= 3 and exceed >= 3 and trend == 2:
            probs = np.array([0.0, 0.0, 0.10, 0.50, 0.40])

        # Rule 3: Dry + any forecast = reduced risk
        elif antecedent == 0 and exceed <= 2:
            probs = np.array([0.55, 0.35, 0.10, 0.0, 0.0])

        # Rule 4: Decreasing trend + low antecedent
        elif trend == 0 and antecedent <= 2 and exceed <= 1:
            probs = np.array([0.65, 0.30, 0.05, 0.0, 0.0])

        # Rule 5: High forecast but dry conditions
        elif antecedent <= 1 and exceed >= 3:
            probs = np.array([0.10, 0.25, 0.45, 0.15, 0.05])

        # Default rules based on base_risk
        elif base_risk < 1:
            probs = np.array([0.50, 0.40, 0.10, 0.0, 0.0])
        elif base_risk < 2:
            probs = np.array([0.10, 0.35, 0.40, 0.15, 0.0])
        elif base_risk < 3:
            probs = np.array([0.05, 0.15, 0.45, 0.30, 0.05])
        elif base_risk < 4:
            probs = np.array([0.0, 0.05, 0.25, 0.50, 0.20])
        else:
            probs = np.array([0.0, 0.0, 0.10, 0.40, 0.50])

        # Apply forecast agreement modifier
        if agreement == 0:  # Low agreement - more uncertainty
            uniform = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
            probs = 0.5 * probs + 0.5 * uniform
        elif agreement == 1:  # Medium agreement
            uniform = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
            probs = 0.8 * probs + 0.2 * uniform

        return probs / probs.sum()

    def _categorize(self, value: float, var: str) -> str:
        """Categorize continuous value to discrete state."""
        if var == 'exceedance':
            if value < 0.2: return 'Very_Low'
            elif value < 0.4: return 'Low'
            elif value < 0.6: return 'Medium'
            elif value < 0.8: return 'High'
            else: return 'Very_High'

        elif var == 'spatial':
            if value < 0.3: return 'Localized'
            elif value < 0.6: return 'Moderate'
            else: return 'Widespread'

        elif var == 'trend':
            trend_map = {
                'decreasing': 'Decreasing',
                'stable': 'Stable',
                'increasing': 'Increasing'
            }
            return trend_map.get(str(value).lower(), 'Stable')

        elif var == 'antecedent':
            antecedent_map = {
                'Dry': 'Dry', 'Normal': 'Normal', 'Wet': 'Wet',
                'Very_Wet': 'Very_Wet', 'Saturated': 'Saturated',
                'Unknown': 'Normal'
            }
            return antecedent_map.get(value, 'Normal')

        elif var == 'agreement':
            if value == 'High': return 'High'
            elif value == 'Medium': return 'Medium'
            else: return 'Low'

        return str(value)

    def process_boundary(self, boundary_data: Dict) -> Dict:
        """Process single boundary through BN inference."""

        # Build evidence
        evidence = {
            'antecedent_rainfall': self._categorize(
                boundary_data.get('antecedent_category', 'Normal'), 'antecedent'
            ),
            'exceedance_prob': self._categorize(
                boundary_data.get('ecmwf_eprob_heavy', 0.0), 'exceedance'
            ),
            'spatial_coverage': self._categorize(
                boundary_data.get('spatial_coverage', 0.0), 'spatial'
            ),
            'rainfall_trend': self._categorize(
                boundary_data.get('rainfall_trend', 'Stable'), 'trend'
            )
        }

        if self.include_agreement_node:
            evidence['forecast_agreement'] = self._categorize(
                boundary_data.get('forecast_agreement', 'Medium'), 'agreement'
            )

        # Run inference
        inference = VariableElimination(self.model)

        risk_query = inference.query(variables=['risk_level'], evidence=evidence)
        risk_probs = risk_query.values
        risk_states = risk_query.state_names['risk_level']
        risk_level = risk_states[np.argmax(risk_probs)]

        action_query = inference.query(variables=['action'], evidence=evidence)
        action_probs = action_query.values
        action_states = action_query.state_names['action']
        recommended_action = action_states[np.argmax(action_probs)]

        return {
            'boundary_id': boundary_data['id'],
            'boundary_name': boundary_data['name'],
            'country': boundary_data['country'],
            'antecedent_rainfall_mm': boundary_data.get('antecedent_rainfall_mm', np.nan),
            'antecedent_category': boundary_data.get('antecedent_category', 'Unknown'),
            'rainfall_trend': boundary_data.get('rainfall_trend', 'Stable'),
            'gefs_eprob_moderate': boundary_data.get('gefs_eprob_moderate', np.nan),
            'ecmwf_eprob_heavy': boundary_data.get('ecmwf_eprob_heavy', np.nan),
            'gefs_eprob_very_heavy': boundary_data.get('gefs_eprob_very_heavy', np.nan),
            'gefs_eprob_extreme': boundary_data.get('gefs_eprob_extreme', np.nan),
            'spatial_coverage_pct': boundary_data.get('spatial_coverage', 0.0) * 100,
            'forecast_agreement': boundary_data.get('forecast_agreement', 'Medium'),
            'evidence': evidence,
            'risk_level': risk_level,
            'risk_probabilities': dict(zip(risk_states, risk_probs)),
            'action_probabilities': dict(zip(action_states, action_probs)),
            'recommended_action': recommended_action,
            'confidence': posterior_confidence(risk_probs)
        }

    def process_all_boundaries(self, boundaries_data: List[Dict]) -> pd.DataFrame:
        """Process all boundaries and return results DataFrame."""
        results = []

        for i, data in enumerate(boundaries_data):
            if (i + 1) % 50 == 0:
                logger.info(f"  Processing {i + 1}/{len(boundaries_data)}...")

            try:
                result = self.process_boundary(data)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process {data.get('id')}: {e}")

        df = pd.DataFrame(results)
        df['high_risk'] = df['recommended_action'].isin(['Prepare', 'Act'])

        return df


# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_flood_risk(
    boundaries_path: str,
    target_date: str,
    imerg_data_dir: str = "./imerg_data",
    gefs_data_dir: str = None,
    obs_days: int = DEFAULT_OBS_DAYS,
    forecast_accum_days: int = 7,
    gefs_run_hour: str = '00',
    output_dir: str = ".",
    include_agreement_node: bool = True
) -> pd.DataFrame:
    """
    Run flood risk analysis for a specific date.

    Args:
        boundaries_path: Path to admin boundaries GeoJSON
        target_date: Analysis date (YYYY-MM-DD)
        imerg_data_dir: Directory with IMERG data
        gefs_data_dir: Directory with GEFS probability data
        obs_days: Observation window in days
        forecast_accum_days: Forecast accumulation period
        gefs_run_hour: GEFS model run hour
        output_dir: Output directory
        include_agreement_node: Include forecast agreement in BN

    Returns:
        DataFrame with flood risk results
    """
    import time
    start = time.time()

    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d')

    logger.info("="*70)
    logger.info(f"FLOOD BN ANALYSIS V1")
    logger.info(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
    logger.info("="*70)

    # Initialize data loader
    loader = FloodDataLoaderV1(
        boundaries_path=boundaries_path,
        imerg_data_dir=imerg_data_dir,
        gefs_data_dir=gefs_data_dir,
        n_obs_days=obs_days
    )

    # Prepare data
    boundaries_data, metadata = loader.prepare_analysis_data(
        target_date=target_date,
        gefs_run_hour=gefs_run_hour,
        forecast_accum_days=forecast_accum_days
    )

    # Initialize BN
    logger.info(f"\nInitializing Bayesian Network...")
    bn = FloodBayesianNetworkV1(include_agreement_node=include_agreement_node)

    # Process boundaries
    logger.info("\nProcessing boundaries...")
    results = bn.process_all_boundaries(boundaries_data)

    elapsed = time.time() - start

    # Summary
    logger.info(f"\n" + "="*70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*70)
    logger.info(f"Target Date: {metadata['target_date']}")
    logger.info(f"Observation Days: {metadata['obs_days_available']}/{metadata['obs_days']}")
    logger.info(f"Forecast Accum: {metadata['forecast_accum_days']} days")
    logger.info(f"Boundaries: {len(results)} | Time: {elapsed:.2f}s")

    logger.info(f"\nAction Distribution:")
    for action in ['Monitor', 'Alert', 'Prepare', 'Act']:
        count = (results['recommended_action'] == action).sum()
        pct = count / len(results) * 100
        logger.info(f"  {action:10}: {count:3} ({pct:.1f}%)")

    # Save results
    output_path = Path(output_dir) / f"flood_bn_v1_{target_date.strftime('%Y%m%d')}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = ['boundary_id', 'boundary_name', 'country',
            'antecedent_rainfall_mm', 'antecedent_category', 'rainfall_trend',
            'ecmwf_eprob_heavy', 'gefs_eprob_very_heavy', 'gefs_eprob_extreme',
            'spatial_coverage_pct', 'forecast_agreement',
            'risk_level', 'recommended_action', 'confidence']
    output_cols = [c for c in cols if c in results.columns]
    results[output_cols].to_csv(output_path, index=False)
    logger.info(f"\nSaved: {output_path}")

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flood BN IBF V1")
    parser.add_argument("--boundaries", required=True, help="Path to boundaries GeoJSON")
    parser.add_argument("--date", required=True, help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--imerg-dir", default="./imerg_data", help="IMERG data directory")
    parser.add_argument("--gefs-dir", default=None, help="GEFS probability data directory")
    parser.add_argument("--obs-days", type=int, default=7, help="Observation window days")
    parser.add_argument("--forecast-accum", type=int, default=7, help="Forecast accumulation days")
    parser.add_argument("--gefs-run", default="00", help="GEFS run hour")
    parser.add_argument("--output-dir", default=".", help="Output directory")

    args = parser.parse_args()

    analyze_flood_risk(
        boundaries_path=args.boundaries,
        target_date=args.date,
        imerg_data_dir=args.imerg_dir,
        gefs_data_dir=args.gefs_dir,
        obs_days=args.obs_days,
        forecast_accum_days=args.forecast_accum,
        gefs_run_hour=args.gefs_run,
        output_dir=args.output_dir
    )
