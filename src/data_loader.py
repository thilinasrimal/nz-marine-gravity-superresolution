import xarray as xr
import rasterio
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import pyproj
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

class NZGravityDataLoader:
    """Data loader for NZ EEZ marine gravity super-resolution project"""
    
    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = Path(config['data']['raw_dir'])
        self.processed_dir = Path(config['data']['processed_dir'])
        self.study_bounds = config['data']['study_bounds']
        self.target_resolution = config['data']['target_resolution_arcmin']
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_sio_gravity(self) -> xr.DataArray:
        """
        Load SIO V33.1 gravity model
        Justification: SIO V33.1 uses deflection of the vertical (DOV) method,
        which is superior for mid-latitude open ocean settings like NZ EEZ
        compared to DTU21's SSH-based method
        """
        path = self.raw_dir / 'grav_33.1.nc'
        ds = xr.open_dataset(path)
        
        # Extract gravity anomaly (typically in mGal)
        # Variable name may vary - adjust as needed
        gravity_var = [v for v in ds.data_vars if 'grav' in v.lower() or 'anomaly' in v.lower()][0]
        gravity = ds[gravity_var]
        
        # Subset to NZ EEZ bounds
        mask_lon = (gravity.lon >= self.study_bounds[0]) & (gravity.lon <= self.study_bounds[2])
        mask_lat = (gravity.lat >= self.study_bounds[1]) & (gravity.lat <= self.study_bounds[3])
        gravity_nz = gravity.where(mask_lon & mask_lat, drop=True)
        
        print(f"SIO Gravity: shape={gravity_nz.shape}, range=[{gravity_nz.min().values:.2f}, {gravity_nz.max().values:.2f}] mGal")
        return gravity_nz
    
    def load_gebco_2026(self) -> xr.DataArray:
        """
        Load GEBCO 2026 bathymetry
        Justification: 2026 release includes updated coastal mapping and
        improved satellite-derived bathymetry in previously data-poor regions
        """
        path = self.raw_dir / 'gebco_2026.nc'
        ds = xr.open_dataset(path)
        
        # GEBCO variable is typically 'elevation' (positive up, negative down)
        bathymetry = -ds['elevation']  # Convert to positive depth
        
        # Subset to NZ EEZ
        mask_lon = (bathymetry.lon >= self.study_bounds[0]) & (bathymetry.lon <= self.study_bounds[2])
        mask_lat = (bathymetry.lat >= self.study_bounds[1]) & (bathymetry.lat <= self.study_bounds[3])
        bathy_nz = bathymetry.where(mask_lon & mask_lat, drop=True)
        
        print(f"GEBCO 2026: shape={bathy_nz.shape}, depth range=[{bathy_nz.min().values:.0f}, {bathy_nz.max().values:.0f}] m")
        return bathy_nz
    
    def load_nz_bathymetry(self) -> np.ndarray:
        """
        Load NZ coastal high-resolution bathymetry (2016)
        Novel contribution: This provides a second bathymetric channel at higher
        resolution than GEBCO, capturing coastal detail that global models miss
        """
        path = self.raw_dir / 'nzbathy_2016.tif'
        
        with rasterio.open(path) as src:
            # Transform to lat/lon if needed
            if src.crs != 'EPSG:4326':
                # Transform coordinates - simplified here
                pass
            
            # Read the data (depth in meters, positive down)
            bathy_nz_highres = src.read(1)
            transform = src.transform
            
            # Get bounds
            left, bottom, right, top = src.bounds
            
        print(f"NZ Bathymetry 2016: shape={bathy_nz_highres.shape}, resolution={src.res}")
        return bathy_nz_highres, (left, right, bottom, top), transform
    
    def load_egm2008(self) -> np.ndarray:
        """
        Load EGM2008 geopotential model
        Used as long-wavelength reference field in Remove-Compute-Restore framework
        """
        path = self.raw_dir / 'EGM2008.tif'
        
        with rasterio.open(path) as src:
            egm = src.read(1)
            
        print(f"EGM2008: shape={egm.shape}")
        return egm, src.transform, src.bounds
    
    def resample_to_common_grid(self, data: xr.DataArray, target_res_deg: float) -> xr.DataArray:
        """Resample all data to common grid resolution"""
        new_lon = np.arange(self.study_bounds[0], self.study_bounds[2], target_res_deg)
        new_lat = np.arange(self.study_bounds[1], self.study_bounds[3], target_res_deg)
        
        resampled = data.interp(lon=new_lon, lat=new_lat, method='bilinear')
        return resampled
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare multi-source input features and target labels
        Input features (4 channels):
        - Channel 1: SIO V33.1 gravity anomaly
        - Channel 2: GEBCO 2026 bathymetry (coarse)
        - Channel 3: NZ Bathymetry 2016 (high-res coastal)
        - Channel 4: EGM2008 residual (after RCR)
        """
        # Load all data sources
        sio_gravity = self.load_sio_gravity()
        gebco_bathy = self.load_gebco_2026()
        nz_bathy, _, _, _ = self.load_nz_bathymetry()
        egm2008, _, _ = self.load_egm2008()
        
        # Resample to common resolution (target resolution for super-resolution output)
        common_res_deg = self.target_resolution / 60  # Convert arc-min to degrees
        sio_resampled = self.resample_to_common_grid(sio_gravity, common_res_deg)
        gebco_resampled = self.resample_to_common_grid(gebco_bathy, common_res_deg)
        
        # For NZ bathymetry and EGM2008, we need to resample to match
        # This requires coordinate alignment - simplified here
        
        # Stack features
        features = np.stack([
            sio_resampled.values,
            gebco_resampled.values,
            # nz_bathy_resampled,  # Add after proper resampling
            # egm2008_resampled,   # Add after proper resampling
        ], axis=0)
        
        # Target is high-resolution gravity (we'll create synthetic targets for training)
        # Using bicubic upsampling of SIO as initial target, refined with shipborne data
        
        return features, features[0:1] * 1.0  # Placeholder target

    def create_patches(self, features: np.ndarray, targets: np.ndarray, 
                       patch_size: int = 64, stride: int = 32) -> list:
        """Create overlapping patches for training"""
        patches = []
        h, w = features.shape[1], features.shape[2]
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                feat_patch = features[:, i:i+patch_size, j:j+patch_size]
                targ_patch = targets[:, i:i+patch_size, j:j+patch_size]
                patches.append((feat_patch, targ_patch))
        
        print(f"Created {len(patches)} patches of size {patch_size}x{patch_size}")
        return patches