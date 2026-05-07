"""
Tsunami simulation around Krakatau using REAL bathymetry + 2D animation.

Bathymetry input options (in order of preference):
  1. --bathy-file PATH     load from a local file (TIFF or NetCDF)
                           use this when NOAA's OPeNDAP server is down.
  2. (default)             stream subset live from NOAA OPeNDAP. Often down;
                           if it 503s, use option 1 instead.
  3. --no-bathy            synthetic continents (offline, less realistic)

Manual download (NOAA OPeNDAP is flaky -- this is more reliable):
    # 60-arc-second GeoTIFF, ~250 MB, plenty of resolution for our grid:
    wget https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif
    python tsunami_paper_compare.py --bathy-file ETOPO_2022_v1_60s_N90W180_surface.tif
    # Or for smaller subsets, use the Grid Extract UI:
    #   https://www.ncei.noaa.gov/maps/grid-extract/

Equations: linear shallow-water with variable bathymetry (Choi et al. 2003,
eqs. 4-6 in flat-Cartesian form, plus f-plane Coriolis). Pseudo-spectral
spatial derivatives. RK2 time stepping. Sponge-layer "reflective" coasts
plus an absorbing edge layer to prevent FFT-periodic boundary reflections.

Install once:
    pip install numpy scipy matplotlib xarray netcdf4 pydap rasterio imageio imageio-ffmpeg

Run:
    python tsunami_paper_compare.py --bathy-file ETOPO_2022_v1_60s_N90W180_surface.tif
    python tsunami_paper_compare.py --no-bathy        # offline mode
"""
