import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from scipy.ndimage import distance_transform_edt, gaussian_filter


# ============================================================
# 1. Parameters
# ============================================================
N_DEFAULT          = 384
L                  = 12000e3
g                  = 9.81
H_DEEP             = 4000.0
H_MIN              = 10.0
T_MAX_DEFAULT      = 14 * 3600
SPONGE_LAND        = 8e-3 * (0.75)
SPONGE_EDGE        = 1.5e-2
EDGE_BUFFER_FRAC   = 0.10

OMEGA_EARTH        = 7.292e-5
F_KRAK             = 2 * OMEGA_EARTH * np.sin(np.deg2rad(-6.1))

R_EARTH            = 6.371e6
KRAK_LAT_DEG       = -6.1
KRAK_LON_DEG       = 105.4
KRAK_LAT           = np.deg2rad(KRAK_LAT_DEG)
KRAK_LON           = np.deg2rad(KRAK_LON_DEG)

# ETOPO 2022 30 arc-sec single global tile via NOAA OPeNDAP. Streams
# subsets so only what we need gets downloaded.
ETOPO_URL = ("https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/30s/"
             "30s_surface_elev_netcdf/ETOPO_2022_v1_30s_N90W180_surface.nc")


# ============================================================
# 2. Stations (paper Tables 1 & 3)
# ============================================================
STATIONS = [
    ("Port Blair",       11.67,  92.75,   4.0,    0.2,   'gauge',  4.5),
    ("Galle",             6.03,  80.22,   5.0,    1.0,   'runup',  4.5),
    ("Negombo",           7.21,  79.84,   7.0,    1.0,   'runup',  5.0),
    ("Bombay",           19.08,  72.87,  11.0,    0.56,  'gauge',  8.6),
    ("Rodriguez Is.",   -19.67,  63.42,   6.5,    1.8,   'runup',  6.67),
    ("Mauritius",       -20.17,  57.50,   6.5,    0.9,   'runup',  7.3),
    ("Cossack",         -20.68, 117.20,   5.25,   1.5,   'runup',  3.0),
    ("Geraldton",       -28.77, 114.60,   9.25,   1.0,   'runup',  4.0)
]


# ============================================================
# 3. Coordinate utilities (azimuthal-equidistant projection)
# ============================================================
def latlon_from_grid(gx, gy, src_x, src_y):
    """Inverse projection: flat-grid (x,y) -> (lat,lon) on the sphere."""
    dx_m = gx - src_x
    dy_m = gy - src_y
    rho = np.sqrt(dx_m**2 + dy_m**2) / R_EARTH
    bearing = np.arctan2(dx_m, dy_m)
    lat = np.arcsin(np.sin(KRAK_LAT) * np.cos(rho)
                    + np.cos(KRAK_LAT) * np.sin(rho) * np.cos(bearing))
    lon = KRAK_LON + np.arctan2(
        np.sin(bearing) * np.sin(rho) * np.cos(KRAK_LAT),
        np.cos(rho) - np.sin(KRAK_LAT) * np.sin(lat))
    return np.rad2deg(lat), np.rad2deg(lon)


def project_station(lat_deg, lon_deg, src_x, src_y):
    """Forward projection: (lat, lon) -> flat-grid (x, y) and arc length."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    cos_c = (np.sin(KRAK_LAT) * np.sin(lat)
             + np.cos(KRAK_LAT) * np.cos(lat) * np.cos(lon - KRAK_LON))
    cos_c = np.clip(cos_c, -1, 1)
    arc_len = R_EARTH * np.arccos(cos_c)
    y_comp = np.sin(lon - KRAK_LON) * np.cos(lat)
    x_comp = (np.cos(KRAK_LAT) * np.sin(lat)
              - np.sin(KRAK_LAT) * np.cos(lat) * np.cos(lon - KRAK_LON))
    bearing = np.arctan2(y_comp, x_comp)
    gx = src_x + arc_len * np.sin(bearing)
    gy = src_y + arc_len * np.cos(bearing)
    return gx, gy, arc_len


# ============================================================
# 4. Bathymetry: ETOPO via OPeNDAP, or synthetic fallback
# ============================================================
def load_etopo_from_file(path):
    """Load ETOPO from a local file. Auto-detects TIFF (uses rasterio) vs
    NetCDF (uses xarray). Returns an xarray DataArray with dims (lat, lon)
    of elevation in meters, ascending in both lat and lon."""
    import xarray as xr
    print(f"Loading bathymetry from {path}...")

    # NetCDF -- try xarray first
    if path.lower().endswith(('.nc', '.netcdf', '.nc4')):
        ds = xr.open_dataset(path)
        # ETOPO files use 'z' for elevation; some derivatives use 'elevation'
        var_name = 'z' if 'z' in ds.variables else list(ds.data_vars)[0]
        da = ds[var_name]
        print(f"  shape: {da.shape}, var: {var_name}")
        return da

    # GeoTIFF -- needs rasterio. ETOPO TIFFs have a single band of int16
    # elevations and proper geo-referencing.
    if path.lower().endswith(('.tif', '.tiff')):
        try:
            import rasterio
        except ImportError:
            raise RuntimeError(
                "Loading GeoTIFF needs rasterio. Install with:\n"
                "    pip install rasterio")
        with rasterio.open(path) as src:
            print(f"  shape: {src.shape}, crs: {src.crs}, "
                  f"bounds: {src.bounds}")
            data = src.read(1)                       # (rows, cols)
            # Build lat/lon coordinate vectors from the affine transform.
            # Rows go top->bottom, so lat[0] is the NORTH edge (largest lat).
            transform = src.transform
            cols = np.arange(src.width)
            rows = np.arange(src.height)
            # Pixel CENTER coordinates
            lons = transform.c + (cols + 0.5) * transform.a
            lats = transform.f + (rows + 0.5) * transform.e
            # Flip rows so lat is ascending (south->north)
            if lats[0] > lats[-1]:
                lats = lats[::-1]
                data = data[::-1, :]
        # Return as a DataArray for uniform downstream handling
        da = xr.DataArray(data, dims=('lat', 'lon'),
                           coords={'lat': lats, 'lon': lons})
        print(f"  lat range: {lats[0]:.2f} to {lats[-1]:.2f}, "
              f"lon range: {lons[0]:.2f} to {lons[-1]:.2f}")
        return da

    raise ValueError(f"Unrecognized file extension: {path}. Use .nc or .tif.")


def load_etopo_subset(cache_path=None):
    """Load ETOPO 2022 30s elevation in a lat/lon box big enough to cover
    the paper's stations.  Returns an xarray DataArray, dims (lat, lon),
    elevation in meters (positive = above sea level)."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached bathymetry: {cache_path}")
        import xarray as xr
        return xr.open_dataarray(cache_path)

    import xarray as xr
    print(f"Streaming ETOPO subset from NOAA (slow on first run)...")
    print(f"  URL: {ETOPO_URL}")

    LAT_MIN, LAT_MAX = -45, 30
    LON_MIN, LON_MAX = 30, 165

    ds = xr.open_dataset(ETOPO_URL)
    sub = ds['z'].sel(lat=slice(LAT_MIN, LAT_MAX),
                       lon=slice(LON_MIN, LON_MAX))
    print(f"  subset shape: {sub.shape} (lat x lon)")
    print(f"  downloading...")
    sub = sub.load()
    print(f"  done. depth range: {float(sub.min())} to {float(sub.max())} m")

    if cache_path:
        sub.to_netcdf(cache_path)
        print(f"  cached: {cache_path}")
    return sub


def project_etopo_to_grid(etopo_da, X_grid, Y_grid, src_x, src_y):
    """Project ETOPO (lat, lon) onto our flat Cartesian grid via inverse
    azimuthal-equidistant projection + bilinear interp. Returns h(x,y)
    where h>0 in water and h<=0 on land."""
    print("Projecting ETOPO onto flat Cartesian grid...")
    lat_grid, lon_grid = latlon_from_grid(X_grid, Y_grid, src_x, src_y)
    lon_grid = ((lon_grid + 180) % 360) - 180

    from scipy.interpolate import RegularGridInterpolator
    lats = etopo_da['lat'].values
    lons = etopo_da['lon'].values
    z = etopo_da.values
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        z = z[::-1, :]

    interp = RegularGridInterpolator(
        (lats, lons), z,
        method='linear', bounds_error=False, fill_value=-H_DEEP)

    pts = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
    elev = interp(pts).reshape(lat_grid.shape)
    h = -elev   # depth positive in water
    return h


def synthetic_bathymetry(X, Y, L_):
    """Fallback: handcrafted continents (offline / --no-bathy)."""
    h = np.full_like(X, H_DEEP)
    src_x, src_y = L_ / 2, L_ / 2
    edge_dist = np.minimum.reduce([X, L_ - X, Y, L_ - Y])
    h = H_MIN + (h - H_MIN) * np.clip(edge_dist / 300e3, 0, 1)
    continents = [
        (src_x - 3000e3, src_y + 3000e3, 1500e3, 1200e3, 400),
        (src_x - 5000e3, src_y - 1000e3, 1200e3, 2500e3, 400),
        (src_x + 3500e3, src_y - 2200e3, 1500e3, 1200e3, 400),
        (src_x + 1500e3, src_y + 3500e3,  600e3,  500e3, 200),
    ]
    for cx, cy, rx, ry, elev in continents:
        d2 = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        h -= (H_DEEP + elev) * np.exp(-d2 * 1.5) * (d2 < 2.0)
    return h


# ============================================================
# 5. Build grid + bathymetry + sponge
# ============================================================
def build_grid_and_bathymetry(N, use_etopo=True, cache_path=None, bathy_file=None):
    dx = L / N
    x  = np.linspace(0, L, N, endpoint=False)
    y  = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')
    src_x, src_y = L / 2, L / 2

    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    k_max = np.max(np.abs(kx))
    dealias = (np.abs(KX) <= (2/3) * k_max) & (np.abs(KY) <= (2/3) * k_max)
    k_norm = np.sqrt(K2) / k_max
    spec_damper = np.exp(-36.0 * k_norm**20)

    if use_etopo:
        try:
            # PRIORITIZE LOCAL FILE: If bathy_file is provided, use it.
            # Only stream if no local file is specified.
            if bathy_file and os.path.exists(bathy_file):
                etopo = load_etopo_from_file(bathy_file)
            else:
                if not bathy_file:
                    print("No local bathy file provided. Attempting stream...")
                else:
                    print(f"Provided file {bathy_file} not found. Attempting stream...")
                etopo = load_etopo_subset(cache_path)
                
            h = project_etopo_to_grid(etopo, X, Y, src_x, src_y)
        except Exception as e:
            print(f"\nBathymetry load failed ({type(e).__name__}: {e})")
            print("Falling back to synthetic bathymetry.\n")
            h = synthetic_bathymetry(X, Y, L)
    else:
        h = synthetic_bathymetry(X, Y, L)

    # Light smoothing prevents spectral ringing at sharp coastlines without
    # blurring shelves.
    h = gaussian_filter(h, sigma=0.7)

    land_mask = h <= 0
    h_water = np.maximum(h, H_MIN)

        # Sponge: was eating the entire continental shelf. Reduce coastal ramp
    # so we only damp very-near-coast cells, leaving shelf for edge waves.
    dist_to_land = distance_transform_edt(~land_mask)
    coast_ramp = np.clip(1.0 - dist_to_land / 1, 0, 1)   # was /4 -- now only 1 cell
    coast_sponge = SPONGE_LAND * (land_mask.astype(float)
                                + coast_ramp * (~land_mask))

    edge_w = max(12, int(EDGE_BUFFER_FRAC * N))
    ii = np.arange(N)[:, None] * np.ones((1, N))
    jj = np.ones((N, 1)) * np.arange(N)[None, :]
    edge_dist = np.minimum.reduce([ii, N - 1 - ii, jj, N - 1 - jj])
    edge_ramp = np.clip(1.0 - edge_dist / edge_w, 0, 1)
    edge_sponge = SPONGE_EDGE * edge_ramp ** 2

    sponge = coast_sponge + edge_sponge
    sponge = gaussian_filter(sponge, sigma=1.0)

    return dict(X=X, Y=Y, dx=dx,
                KX=KX, KY=KY, K2=K2,
                dealias=dealias, spec_damper=spec_damper,
                h_water=h_water, land_mask=land_mask, sponge=sponge,
                src_x=src_x, src_y=src_y)


# ============================================================
# 6. Station -> grid
# ============================================================
def find_water_cell(i, j, land_mask, N):
    if 0 <= i < N and 0 <= j < N and not land_mask[i, j]:
        return i, j
    for r in range(1, 30):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if abs(di) != r and abs(dj) != r:
                    continue
                ii, jj = i + di, j + dj
                if (0 <= ii < N and 0 <= jj < N
                        and not land_mask[ii, jj]):
                    return ii, jj
    return None


# ============================================================
# 7. Simulation: gauges + spatial snapshots
# ============================================================
def simulate(gd, station_ij, N, T_MAX, n_snapshots=200):
    X, Y = gd['X'], gd['Y']
    KX, KY = gd['KX'], gd['KY']
    dealias, spec_damper = gd['dealias'], gd['spec_damper']
    h_water, land_mask, sponge = gd['h_water'], gd['land_mask'], gd['sponge']
    src_x, src_y, dx = gd['src_x'], gd['src_y'], gd['dx']

    c_max = np.sqrt(g * h_water.max())
    dt = 0.3 * dx / c_max
    steps = int(T_MAX / dt)
    snap_every = max(1, steps // n_snapshots)
    print(f"Grid: {N}x{N}, dx={dx/1e3:.1f} km, dt={dt:.1f} s, "
          f"steps={steps}, c_max={c_max:.0f} m/s, T={T_MAX/3600:.0f}h")
    print(f"Snapshots: every {snap_every} steps "
          f"(~{steps//snap_every} animation frames)")

    def ddx(f):
        return np.fft.ifft2(1j * KX * np.fft.fft2(f) * dealias).real
    def ddy(f):
        return np.fft.ifft2(1j * KY * np.fft.fft2(f) * dealias).real
    def filt(f):
        return np.fft.ifft2(np.fft.fft2(f) * spec_damper).real

    def rhs(eta, M, Nf):
        deta = -(ddx(M) + ddy(Nf))
        dM = -g * h_water * ddx(eta) + F_KRAK * Nf
        dN = -g * h_water * ddy(eta) - F_KRAK * M
        deta[land_mask] = 0.0
        return deta, dM, dN

    # Move the source into open Indian Ocean, southwest of the strait, to
    # avoid Sumatra/Java cutting off most of the radiation. Krakatau is at
    # (src_x, src_y); we shift to ~150 km SW into deep water.
    src_x_eff = src_x - 150e3       # west
    src_y_eff = src_y - 150e3       # south
    print(f"  Source shifted SW into open ocean: "
        f"({src_x_eff/1e3:.0f}, {src_y_eff/1e3:.0f}) km")

    # Sanity check the new location
    i_eff = int(round(src_y_eff / dx))
    j_eff = int(round(src_x_eff / dx))
    print(f"  depth at shifted source: {h_water[i_eff, j_eff]:.0f} m, "
        f"land cells in 10-cell box: "
        f"{land_mask[i_eff-10:i_eff+11, j_eff-10:j_eff+11].sum()}/441")

    # ---- Initial condition: square depression matching paper's geometry ----
# Paper Sect. 4: 7.2 x 7.2 km flat square, depth 222 m, vol 11.5 km^3.
# At dx=31 km we can't resolve 7.2 km, so we use a square that's at least
# 2 cells wide and scale the depth to preserve the displaced volume.
    SRC_VOLUME_TARGET = 11.5e9
    src_half = max(2 * dx, 3.6e3)              # at least 2 cells half-width
    side = 2 * src_half                         # full side
    src_amp = 2*SRC_VOLUME_TARGET / (side ** 2)   # depth s.t. amp*side^2 = vol

    # Soft-edged square: flat top across the interior, tanh shoulders one
    # cell wide so the FFT doesn't ring at the edges.
    fall = dx
    rx = np.abs(X - src_x_eff)
    ry = np.abs(Y - src_y_eff)
    sx = 0.5 * (1 - np.tanh((rx - src_half) / fall))
    sy = 0.5 * (1 - np.tanh((ry - src_half) / fall))
    eta = -src_amp * sx * sy

    vol_before = -np.sum(eta) * dx * dx
    eta[land_mask] = 0.0
    vol_after = -np.sum(eta) * dx * dx
    print(f"  Volume before land mask: {vol_before/1e9:.2f} km^3")
    print(f"  Volume after  land mask: {vol_after/1e9:.2f} km^3 "
        f"({100*vol_after/vol_before:.0f}% retained)")

    eta = filt(eta)
    M  = np.zeros_like(eta)
    Nf = np.zeros_like(eta)

    init_vol = -np.sum(eta) * dx * dx

    gauges = {k: [] for k, ij in enumerate(station_ij) if ij is not None}
    times = []
    snapshots = []
    snap_times = []

    print("Simulating...")
    for step in range(1, steps + 1):
        k1e, k1M, k1N = rhs(eta, M, Nf)
        k2e, k2M, k2N = rhs(eta + 0.5*dt*k1e,
                            M   + 0.5*dt*k1M,
                            Nf  + 0.5*dt*k1N)
        eta = eta + dt * k2e
        M   = M   + dt * k2M
        Nf  = Nf  + dt * k2N
        M  = M  / (1 + sponge * dt)
        Nf = Nf / (1 + sponge * dt)
        eta = eta / (1 + 0.3 * sponge * dt)
        eta = filt(eta); M = filt(M); Nf = filt(Nf)
        eta[land_mask] = 0.0

        t_now = step * dt
        times.append(t_now)
        for k, ij in enumerate(station_ij):
            if ij is None:
                continue
            i, j = ij
            gauges[k].append(eta[i, j])

        if step % snap_every == 0:
            snapshots.append(eta.copy())
            snap_times.append(t_now)

        if step % 200 == 0:
            print(f"  step {step}/{steps}  t={t_now/3600:.2f} h  "
                  f"max|eta|={np.max(np.abs(eta)):.2f} m")

    return (np.array(times),
            {k: np.array(v) for k, v in gauges.items()},
            snapshots, np.array(snap_times))


# ============================================================
# 8. Arrival / amplitude analysis
# ============================================================
def analyze_gauge(t, eta_ts):
    abs_eta = np.abs(eta_ts)
    peak = abs_eta.max()
    if peak < 0.005:                # was 0.02
        return None, peak
    threshold = max(0.005, 0.10 * peak)   # was 0.02
    arrival_idx = np.argmax(abs_eta > threshold)
    return t[arrival_idx] / 3600, peak


# ============================================================
# 9. Plot: gauge time series
# ============================================================
def plot_gauges(t, gauges, station_ij, save_path=None):
    active = [k for k, ij in enumerate(station_ij) if ij is not None]
    ncols = 2
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.0 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    t_h = t / 3600

    # Compute a sensible shared y-range. Use the 99th percentile of the
    # largest gauge so an outlier spike doesn't squash everything else.
    all_peaks = [np.abs(gauges[k]).max() for k in active]
    y_lim = max(all_peaks) * 1.1               # 10% headroom
    # If you want detail on quiet stations to be visible too, use this
    # instead: a percentile so outliers don't dominate.
    # y_lim = np.percentile([np.abs(gauges[k]) for k in active], 99) * 1.5

    for idx, k in enumerate(active):
        ax = axes[idx]
        name, lat, lon, tt_obs, amp_obs, kind, tt_paper = STATIONS[k]
        ax.plot(t_h, gauges[k], 'k-', linewidth=0.6)
        ax.axvline(tt_obs, color='red', linestyle='--', linewidth=0.8,
                   label=f'tt_obs={tt_obs}h')
        ax.axvline(tt_paper, color='blue', linestyle=':', linewidth=0.8,
                   label=f'tt_paper={tt_paper}h')
        tt_sim, amp_sim = analyze_gauge(t, gauges[k])
        if tt_sim is not None:
            ax.axvline(tt_sim, color='green', linestyle='-', linewidth=1.0,
                       alpha=0.7, label=f'tt_sim={tt_sim:.2f}h')
        ax.axhline(amp_obs, color='orange', linestyle=':', linewidth=0.5)
        ax.axhline(-amp_obs, color='orange', linestyle=':', linewidth=0.5)
        # Annotate the simulated peak amplitude in the corner
        ax.text(0.02, 0.95, f'sim peak: {amp_sim:.3f} m',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='gray', alpha=0.8))
        ax.set_title(f'{name}  ({lat:+.1f}°, {lon:+.1f}°)  '
                     f'amp_obs={amp_obs}m ({kind})', fontsize=9)
        ax.set_ylabel('η (m)', fontsize=8)
        ax.set_ylim(-y_lim, y_lim)             # SHARED Y-RANGE
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)
    for ax in axes[len(active):]:
        ax.set_visible(False)
    for ax in axes[max(0, len(active) - ncols):len(active)]:
        ax.set_xlabel('time (hours since source)', fontsize=9)
    fig.suptitle('Simulated tide-gauge time series @ paper stations',
                 fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved gauge plot: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ============================================================
# 10. 2D propagation animation
# ============================================================
def render_animation(snapshots, snap_times, gd, station_ij, save_path,
                     fps=15):
    """Animate η(x,y,t). Land in brown, water in red/blue diverging,
    stations marked yellow, source as a red star."""
    print(f"Rendering animation -> {save_path} ...")
    X, Y = gd['X'], gd['Y']
    land_mask = gd['land_mask']

    late = snapshots[len(snapshots) // 4:]
    vmax = 0.05

    fig, ax = plt.subplots(figsize=(9, 8), facecolor='black')
    ax.set_facecolor('#0a0a14')
    extent = [0, L / 1e3, 0, L / 1e3]

    im = ax.imshow(snapshots[0], origin='lower', extent=extent,
                   cmap='seismic', vmin=-vmax, vmax=vmax,
                   interpolation='bilinear', animated=True)

    land_rgba = np.zeros((*land_mask.shape, 4))
    land_rgba[land_mask] = [0.45, 0.32, 0.20, 1.0]
    ax.imshow(land_rgba, origin='lower', extent=extent, interpolation='nearest')
    ax.contour(X / 1e3, Y / 1e3, land_mask.astype(float),
               levels=[0.5], colors='black', linewidths=0.4)

    for k, ij in enumerate(station_ij):
        if ij is None:
            continue
        name = STATIONS[k][0]
        i, j = ij
        gy_km = (i + 0.5) * gd['dx'] / 1e3
        gx_km = (j + 0.5) * gd['dx'] / 1e3
        ax.plot(gx_km, gy_km, 'o', markersize=5,
                markerfacecolor='yellow', markeredgecolor='black',
                markeredgewidth=0.5)
        ax.annotate(name, (gx_km, gy_km), fontsize=7, color='black',
                    xytext=(4, 4), textcoords='offset points')

    src_x, src_y = gd['src_x'] / 1e3, gd['src_y'] / 1e3
    ax.plot(src_x, src_y, '*', markersize=12, color='red',
            markeredgecolor='white', markeredgewidth=0.8)

    ax.set_xlabel('x (km)', color='white')
    ax.set_ylabel('y (km)', color='white')
    ax.tick_params(colors='white')
    title = ax.set_title(f'Tsunami propagation, t = 0.00 h',
                         color='white', fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('η (m)', color='white')
    cbar.ax.tick_params(colors='white')
    cbar.outline.set_edgecolor('white')

    def update(frame_idx):
        im.set_data(snapshots[frame_idx])
        title.set_text(
            f'Tsunami propagation, t = {snap_times[frame_idx]/3600:.2f} h')
        return [im, title]

    anim = manim.FuncAnimation(
        fig, update, frames=len(snapshots),
        interval=1000 / fps, blit=False)

    try:
        writer = manim.FFMpegWriter(fps=fps, bitrate=2000, codec='libx264')
        anim.save(save_path, writer=writer, dpi=110,
                  savefig_kwargs={'facecolor': 'black'})
    except Exception as e:
        print(f"  ffmpeg failed ({e}); trying pillow GIF...")
        gif_path = save_path.rsplit('.', 1)[0] + '.gif'
        anim.save(gif_path, writer='pillow', fps=fps)
        save_path = gif_path
    plt.close(fig)
    print(f"  saved: {save_path}")


# ============================================================
# 11. Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('-N', type=int, default=N_DEFAULT)
    p.add_argument('--hours', type=float, default=T_MAX_DEFAULT/3600)
    p.add_argument('--no-bathy', action='store_true',
                   help='skip ETOPO download, use synthetic bathymetry')
    p.add_argument('--cache', type=str, default='etopo_subset.nc',
                   help='local cache file for ETOPO subset (empty to disable)')
    
    # 1. THIS TELLS THE SCRIPT TO LOOK FOR YOUR FLAG
    p.add_argument('--bathy-file', type=str, default=None,
                   help='load from a local file (TIFF or NetCDF)')
                   
    p.add_argument('--save-plot', type=str, default='gauges.png')
    p.add_argument('--save-anim', type=str, default='tsunami.mp4',
                   help='MP4 of the wave; empty string to skip')
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()

    N = args.N
    T_MAX = args.hours * 3600

    cache = args.cache if args.cache else None
    
    # 2. THIS PASSES THE FLAG INTO THE BUILDER
    gd = build_grid_and_bathymetry(
        N, use_etopo=(not args.no_bathy), cache_path=cache, bathy_file=args.bathy_file)

    i_src = int(round(gd['src_y'] / gd['dx']))
    j_src = int(round(gd['src_x'] / gd['dx']))
    print(f"\nBathymetry sanity check (source at i={i_src}, j={j_src}):")
    print(f"  depth at source: {gd['h_water'][i_src, j_src]:.1f} m")
    print(f"  depths in 5-cell radius: "
      f"min={gd['h_water'][i_src-5:i_src+6, j_src-5:j_src+6].min():.1f}, "
      f"max={gd['h_water'][i_src-5:i_src+6, j_src-5:j_src+6].max():.1f}")
    print(f"  land cells in 10-cell box: "
      f"{gd['land_mask'][i_src-10:i_src+11, j_src-10:j_src+11].sum()} / 441")
        
    src_x, src_y = gd['src_x'], gd['src_y']

    print("\nStation placement (great-circle from Krakatau):")
    station_ij = []
    station_arc = []
    for name, lat, lon, *_ in STATIONS:
        gx, gy, arc = project_station(lat, lon, src_x, src_y)
        i = int(round(gy / gd['dx']))
        j = int(round(gx / gd['dx']))
        if not (0 <= i < N and 0 <= j < N):
            print(f"  {name:18s} arc={arc/1e3:6.0f} km   OUT OF DOMAIN")
            station_ij.append(None)
            station_arc.append(arc)
            continue
        result = find_water_cell(i, j, gd['land_mask'], N)
        if result is None:
            print(f"  {name:18s} arc={arc/1e3:6.0f} km   no water near")
            station_ij.append(None)
        else:
            station_ij.append(result)
            i2, j2 = result
            note = "" if (i2, j2) == (i, j) else f" (nudged from {i},{j})"
            print(f"  {name:18s} arc={arc/1e3:6.0f} km   "
                  f"grid=({i2},{j2}){note}")
        station_arc.append(arc)

    t, gauges, snapshots, snap_times = simulate(
        gd, station_ij, N, T_MAX)

    print("\n" + "=" * 96)
    print(f"{'Station':<18}{'arc(km)':>9}{'tt_obs':>9}{'tt_paper':>10}"
          f"{'tt_sim':>9}{'amp_obs':>10}{'kind':>7}{'amp_sim':>10}{'note':>20}")
    print("-" * 96)
    for k, (name, lat, lon, tt_obs, amp_obs, kind, tt_paper) in enumerate(STATIONS):
        arc_km = station_arc[k] / 1e3
        if station_ij[k] is None:
            print(f"{name:<18}{arc_km:>9.0f}{tt_obs:>9.2f}{tt_paper:>10.2f}"
                  f"{'--':>9}{amp_obs:>10.2f}{kind:>7}{'--':>10}"
                  f"{'out of domain':>20}")
            continue
        tt_sim, amp_sim = analyze_gauge(t, gauges[k])
        tt_str = f"{tt_sim:.2f}" if tt_sim else "n/a"
        amp_str = f"{amp_sim:.3f}"
        print(f"{name:<18}{arc_km:>9.0f}{tt_obs:>9.2f}{tt_paper:>10.2f}"
              f"{tt_str:>9}{amp_obs:>10.2f}{kind:>7}{amp_str:>10}{'':>20}")
    print("=" * 96)

    if not args.no_plot:
        plot_gauges(t, gauges, station_ij, save_path=args.save_plot)

    if args.save_anim:
        render_animation(snapshots, snap_times, gd, station_ij,
                         save_path=args.save_anim)


if __name__ == '__main__':
    main()