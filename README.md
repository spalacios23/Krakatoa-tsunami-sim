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

Produces `gauges.png` (tide-gauge time series at the paper's stations)
and `tsunami.mp4` (2D wave-field animation).

Useful flags: `-N 256` for a coarser/faster grid, `--hours 8` for a
shorter sim, `--no-plot` or `--save-anim ""` to skip outputs.

## Outputs

| File | What |
|---|---|
| `gauges.png` | η(t) at 8 stations from the paper, with observed and simulated arrival times overlaid |
| `tsunami.mp4` | 2D animation of η(x,y,t) across the Indian Ocean |
| stdout | Comparison table: observed vs paper-simulated vs our-simulated arrival times and amplitudes |

## What works, what doesn't

Travel times for Indian Ocean stations within ~3000 km (Galle, Cossack,
Rodriguez, Mauritius) match the paper's spherical-grid simulation to
within 10–15%. Amplitudes are ~5–10× lower than paper values — fundamental
cost of the flat-Cartesian projection and 31 km grid resolution
(paper uses 3.7 km). "Dark zone" stations on the back side of continents
(Bombay, Negombo) require resolved continental shelves to capture edge-wave
propagation, which our grid is too coarse for. 

## Physics

`η, M, N` (sea-surface displacement and volume fluxes) evolve under:

- pressure-gradient forcing `g · h(x,y) · ∇η` with bathymetry-dependent
  wave speed `c = √(g h)`
- f-plane Coriolis at Krakatau's latitude
- spectral filtering for Gibbs suppression
- 2/3-rule dealiasing during pseudo-spectral derivatives
- sponge damping at coasts (fake reflection) and outer edges (absorber)

Source: a sub-grid square depression carrying 11.5 km³ of displaced
volume per the paper, shifted ~150 km SW into the open Indian Ocean
because the Sunda Strait is sub-grid at our resolution.
