# Krakatau Tsunami Simulation

A pseudo-spectral shallow-water simulation of the 1883 Krakatau tsunami,
reproducing the setup of [Choi et al. (2003)][1]. Linear shallow-water
equations on a flat-Cartesian grid with real ETOPO bathymetry, FFT-based
spatial derivatives, RK2 time stepping, sponge-layer reflective coasts,
and an absorbing edge buffer.

[1]: https://nhess.copernicus.org/articles/3/321/2003/

## Setup

```bash
python3 -m venv tsunami-env
source tsunami-env/bin/activate
pip install numpy scipy matplotlib xarray rasterio imageio imageio-ffmpeg

# bathymetry, ~250 MB, run once
wget https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif
```

## Run

```bash
python tsunami_sim_local.py --bathy-file ETOPO_2022_v1_60s_N90W180_surface.tif
```


