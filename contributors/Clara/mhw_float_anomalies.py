"""
mhw_float_anomalies.py — CRV.
Match every processed BGC-Argo profile to the SOCA
monthly climatology and turn each one into a single anomaly value, then plot
those anomalies as a time series per marine heatwave box.

[Runs on the parquet written by mhw_float_processing.py.]

ANOMALY DEFINITION:
    anomaly = (float profile integrated 0-200 m)
            - (climatology profile integrated 0-200 m, for the calendar month
               of that profile, at the nearest climatology cell)
Both sides are integrated over the SAME uniform depth grid before differencing,
so the two integrals cover identical depths and the difference is not a partial
column minus a full one. A profile that does not actually sample the column is
dropped rather than integrated over whatever it happens to have: it needs a
level at or above TOP_MAX, a level at or below 200 - DZ, and no internal gap
wider than GAP_MAX.

MATCHING
Nearest climatology cell by KDTree on (lon, lat), in the same spirit as
MHW_funcs.get_oisst_flag. Longitudes go through a 0-360 wrap first when the box
crosses the dateline, so a profile at 179 E is not matched to a cell at 179 W
across a 358-degree gap. The climatology is box-subset before the tree is
built, which keeps a global 3 GB file off the critical path.


NOTES: the SOCA climatology's reference period is 1998-2023, which includes
both heatwaves. The baseline is pulled toward the event, so these anomalies are
damped rather than inflated.

chl and bbp come straight from the climatology. Cphyto and POC are computed
FROM the climatology's own bbp with the same functions applied to the float
data, so each anomaly is a like-for-like difference rather than a comparison
across two different carbon algorithms. Note the climatology carries a native
POC variable that this file does not contain -- see the note in
mhw_float_processing.POC_RELATIONS.

Usage:
    python mhw_float_anomalies.py                     # both boxes
    python mhw_float_anomalies.py --events NA2023
    python mhw_float_anomalies.py --zmax 100
    python mhw_float_anomalies.py --data-dir /path/to/data --clim-file /path/to/clim.nc

Requires mhw_float_processing.py in the same folder (it is imported), the
parquet that script writes, and the SOCA climatology netCDF. See README.md.
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree

# np.trapz was renamed np.trapezoid in numpy 2.0 and removed from 2.x. Bind
# whichever exists so this runs on both.
trapezoid = getattr(np, "trapezoid", None) or np.trapz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Imported as a module, not as values: --data-dir rebinds paths inside it at
# runtime, and `from ... import HERE` would freeze a stale copy here.
import mhw_float_processing as mfp
from mhw_float_processing import EVENTS, bbp_to_cphyto, bbp_to_poc, load_mhw_boxes

# SOCA monthly climatology (CMEMS). Resolution order: --clim-file, then the
# MHW_SOCA_CLIM environment variable, then <data-dir>/SOCA_monthly_clim_chl_bbp_global.nc
CLIM_FILE = os.environ.get("MHW_SOCA_CLIM")
FIG_DIR = ANOM_DIR = None       # filled by set_paths()


def set_paths(data_dir=None, clim_file=None):
    """Point the module at a data root and a climatology file."""
    global CLIM_FILE, FIG_DIR, ANOM_DIR
    mfp.set_data_dir(data_dir)
    if clim_file:
        CLIM_FILE = os.path.abspath(os.path.expanduser(clim_file))
    elif not CLIM_FILE:
        CLIM_FILE = os.path.join(mfp.HERE, "SOCA_monthly_clim_chl_bbp_global.nc")
    FIG_DIR = os.path.join(mfp.HERE, "figures", "anomalies_matchup")
    ANOM_DIR = os.path.join(mfp.HERE, "anomalies")

# integration grid and the coverage a profile must have to be integrated at all
DZ = 5.0
ZMAX = 200.0                     # integration depth; --zmax overrides
ZGRID = np.arange(0.0, ZMAX + DZ, DZ)
TOP_MAX = 10.0        # shallowest level must be at least this shallow
GAP_MAX = 25.0        # widest internal gap tolerated, m
MAX_MATCH_DEG = 1.0   # reject a matchup further than this from a climatology cell

# float column -> (climatology column, label, units of the 0-200 m integral)
VARIABLES = {
    "CHLA": ("chl", "chl", "mg m$^{-2}$"),
    "BBP700": ("bbp", "b$_{bp}$(700)", "m$^{-1}$ m"),
    "CPHYTO": ("cphyto", "C$_{phyto}$", "mg C m$^{-2}$"),
    "POC": ("poc", "POC", "mg C m$^{-2}$"),
}


# --------------------------------------------------------------------------- integration
def set_zmax(zmax):
    """Reset the integration grid. The climatology, not the floats, is what
    limits how deep this can go -- see load_climatology()."""
    global ZMAX, ZGRID
    ZMAX = float(zmax)
    ZGRID = np.arange(0.0, ZMAX + DZ, DZ)


def integrate_profile(pres, values, zgrid=None):
    """Trapezoidal 0-200 m integral of one profile, or NaN if it does not span
    the column. Values are binned to zgrid first so that a profile sampling
    every 2 m near the surface does not out-weight one sampling every 10 m."""
    zgrid = ZGRID if zgrid is None else zgrid
    ok = np.isfinite(pres) & np.isfinite(values)
    if ok.sum() < 4:
        return np.nan
    z, v = pres[ok], values[ok]
    order = np.argsort(z)
    z, v = z[order], v[order]
    z, v = z[z <= zgrid[-1]], v[z <= zgrid[-1]]
    if z.size < 4 or z[0] > TOP_MAX or z[-1] < zgrid[-1] - DZ:
        return np.nan
    if np.diff(z).max() > GAP_MAX:
        return np.nan
    binned = pd.Series(v).groupby(
        zgrid[np.clip(np.digitize(z, (zgrid[:-1] + zgrid[1:]) / 2.0), 0, len(zgrid) - 1)]
    ).median()
    return float(trapezoid(binned.values, binned.index.values))


def integrate_floats(df):
    """One row per profile: position, date, and the 0-200 m integral of each variable."""
    rows = []
    for (wmo, cycle), prof in df.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"], sort=False):
        rec = dict(PLATFORM_NUMBER=wmo, CYCLE_NUMBER=cycle,
                   JULD=prof["JULD"].mean(),
                   LATITUDE=prof["LATITUDE"].mean(),
                   LONGITUDE=prof["LONGITUDE"].mean(),
                   N_LEVELS=len(prof))
        pres = prof["PRES"].values
        for var in VARIABLES:
            rec[f"{var}_INT"] = integrate_profile(pres, prof[var].values)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["MONTH"] = out["JULD"].dt.month
    out["YEAR"] = out["JULD"].dt.year
    return out.sort_values("JULD").reset_index(drop=True)


# --------------------------------------------------------------------------- climatology
def load_climatology(box, poc_relation="johnson"):
    """SOCA monthly climatology cut to the box, with Cphyto and POC derived
    from its own bbp, integrated 0-200 m for every cell and month.

    Returns (integrals, lons, lats) where integrals[var] has shape
    (12, n_lat, n_lon). Integrating the climatology once per cell up front is
    what makes the per-profile matchup cheap: the alternative, integrating a
    fresh column for each of several thousand profiles, re-reads the same
    cells over and over.
    """
    ds = xr.open_dataset(CLIM_FILE)[["chl", "bbp"]]
    sel = dict(latitude=slice(box["lat0"] - 1, box["lat1"] + 1),
               depth=slice(0, ZGRID[-1]))
    if box["lon0"] <= box["lon1"]:
        sub = ds.sel(longitude=slice(box["lon0"] - 1, box["lon1"] + 1), **sel).load()
    else:
        # dateline: take the two halves, then glue them on a continuous 0-360 axis
        east = ds.sel(longitude=slice(box["lon0"] - 1, 180.0), **sel).load()
        west = ds.sel(longitude=slice(-180.0, box["lon1"] + 1), **sel).load()
        east = east.assign_coords(longitude=east.longitude % 360)
        west = west.assign_coords(longitude=west.longitude % 360)
        sub = xr.concat([east, west], dim="longitude").sortby("longitude")
    ds.close()

    sub = sub.interp(depth=ZGRID)

    # SOCA columns carry interior depth gaps: in the NA2023 box 54% of cells
    # have at least one NaN level and 17% are empty top to bottom. A trapezoid
    # over a column with a single NaN returns NaN, so integrating the raw
    # columns silently threw away more than half the matchups. Interior gaps
    # are interpolated in depth; leading/trailing gaps are deliberately NOT
    # filled, so a column that simply stops at 150 m still yields NaN rather
    # than a fabricated 0-200 m integral.
    sub = sub.interpolate_na(dim="depth", method="linear", use_coordinate=True)

    # THE BINDING CONSTRAINT IS THE CLIMATOLOGY'S DEPTH REACH, NOT THE FLOATS.
    # SOCA's valid depth shortens with depth: in the NA2023 box 17% of cells
    # are empty top to bottom (land / no retrieval) but the NaN fraction climbs
    # from 18% at 100 m to 32% at 150 m to 54% at 200 m. Those are TRAILING
    # gaps, so interpolation cannot fill them and must not try. Integrating to
    # 200 m therefore discards more than half the North Atlantic box; --zmax
    # 100 roughly doubles the usable sample at the cost of the 100-200 m layer.
    nan_by_depth = np.isnan(sub["chl"].values).mean(axis=(0, 2, 3))
    usable = 1.0 - float(np.isnan(sub["chl"].values).any(axis=1).mean())
    print(f"  climatology depth reach: {100*nan_by_depth[0]:.0f}% of cells empty at "
          f"the surface, {100*nan_by_depth[-1]:.0f}% at {ZMAX:.0f} m "
          f"-> {100*usable:.0f}% of cells give a full 0-{ZMAX:.0f} m integral")

    sub["cphyto"] = bbp_to_cphyto(sub["bbp"])
    sub["poc"] = (("month", "depth", "latitude", "longitude"),
                  bbp_to_poc(sub["bbp"].values, relation=poc_relation))

    integrals = {}
    for _, (clim_var, _, _) in VARIABLES.items():
        integrals[clim_var] = trapezoid(
            sub[clim_var].transpose("month", "latitude", "longitude", "depth").values,
            ZGRID, axis=-1)
    return integrals, sub["longitude"].values, sub["latitude"].values


def match_climatology(prof, integrals, clim_lons, clim_lats, wrap):
    """Nearest climatology cell per profile, then its integral for that month.

    KDTree over the box's cell centres, as in MHW_funcs.get_oisst_flag. A
    profile further than MAX_MATCH_DEG from any cell gets NaN rather than a
    silently wrong match -- that happens over land or at a box edge.
    """
    grid_lon, grid_lat = np.meshgrid(clim_lons, clim_lats)
    tree = KDTree(np.c_[grid_lon.ravel(), grid_lat.ravel()])

    query_lon = prof["LONGITUDE"].values % 360 if wrap else prof["LONGITUDE"].values
    dist, idx = tree.query(np.c_[query_lon, prof["LATITUDE"].values])
    row, col = np.unravel_index(idx, grid_lon.shape)
    month_idx = prof["MONTH"].values - 1
    too_far = dist > MAX_MATCH_DEG

    out = prof.copy()
    out["MATCH_DIST_DEG"] = dist
    for var, (clim_var, _, _) in VARIABLES.items():
        clim_int = integrals[clim_var][month_idx, row, col].astype(float)
        clim_int[too_far] = np.nan
        out[f"{var}_CLIM"] = clim_int
        out[f"{var}_ANOM"] = out[f"{var}_INT"] - clim_int
    if too_far.any():
        print(f"    {int(too_far.sum())} profiles had no climatology cell within "
              f"{MAX_MATCH_DEG} deg and were left as NaN")
    return out


# --------------------------------------------------------------------------- plots
def plot_anomaly_timeseries(prof, event, box, out_dir):
    """One panel per variable: the per-profile anomaly, its monthly median,
    and the MHW window shaded."""
    fig, axes = plt.subplots(len(VARIABLES), 1, figsize=(13, 2.9 * len(VARIABLES)),
                             sharex=True)
    for ax, (var, (_, label, units)) in zip(axes, VARIABLES.items()):
        col = f"{var}_ANOM"
        data = prof.dropna(subset=[col])
        ax.axvspan(box["date_start"], box["date_end"], color="0.87", zorder=0)
        ax.axhline(0, color="k", lw=0.8, zorder=1)
        ax.scatter(data["JULD"], data[col], s=7, c="#2c7fb8", alpha=0.35,
                   linewidths=0, zorder=2)

        monthly = data.set_index("JULD")[col].resample("MS").median()
        centre = monthly.index + pd.Timedelta(days=15)
        ax.plot(centre, monthly.values, color="k", lw=1.8, zorder=4)
        ax.fill_between(centre, 0, monthly.values, where=monthly.values >= 0,
                        color="crimson", alpha=0.35, linewidth=0, interpolate=True, zorder=3)
        ax.fill_between(centre, 0, monthly.values, where=monthly.values < 0,
                        color="steelblue", alpha=0.35, linewidth=0, interpolate=True, zorder=3)

        finite = np.abs(data[col].values)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            lim = np.percentile(finite, 97.5) or finite.max()
            if lim > 0:
                ax.set_ylim(-lim, lim)
        ax.set_ylabel(f"{label} anomaly\n({units})", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    n = int(prof["CHLA_ANOM"].notna().sum())
    fig.text(0.005, 0.995,
             f"{event}   0-{ZMAX:.0f} m integrated anomaly vs SOCA monthly climatology   "
             f"{n} profiles, {prof.PLATFORM_NUMBER.nunique()} floats   "
             f"(grey band = MHW window)", fontsize=12, fontweight="bold", va="top")
    fig.subplots_adjust(hspace=0.16)
    out = os.path.join(out_dir,
                       f"{event}_integrated_anomaly_timeseries_{ZMAX:.0f}m.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", nargs="*", default=None, help="subset of %s" % list(EVENTS))
    parser.add_argument("--poc", default="johnson")
    parser.add_argument("--data-dir", default=None,
                        help="folder holding processed/ (default: $MHW_FLOAT_DATA, "
                             "else this script's folder)")
    parser.add_argument("--clim-file", default=None,
                        help="SOCA monthly climatology netCDF (default: "
                             "$MHW_SOCA_CLIM, else <data-dir>/SOCA_monthly_clim_chl_bbp_global.nc)")
    parser.add_argument("--zmax", type=float, default=200.0,
                        help="integration depth in m (default 200; the SOCA "
                             "climatology only reaches 200 m in ~46%% of the "
                             "NA2023 box, so 100 gives a much larger sample)")
    args = parser.parse_args()
    set_zmax(args.zmax)
    set_paths(args.data_dir, args.clim_file)
    if not os.path.exists(CLIM_FILE):
        raise SystemExit(
            f"SOCA climatology not found at {CLIM_FILE}\n"
            f"Pass --clim-file, or set $MHW_SOCA_CLIM. See README.md for how to "
            f"download it from CMEMS.")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(ANOM_DIR, exist_ok=True)

    for event, box in load_mhw_boxes(args.events).items():
        print(f"\n=== {event}", flush=True)
        path = os.path.join(mfp.OUT_DIR, f"{event}_profiles.parquet")
        if not os.path.exists(path):
            print(f"  {path} missing -- run mhw_float_processing.py first")
            continue
        df = pd.read_parquet(path)

        prof = integrate_floats(df)
        print(f"  {len(prof)} profiles from {prof.PLATFORM_NUMBER.nunique()} floats; "
              f"0-200 m integral available for "
              f"{int(prof.CHLA_INT.notna().sum())} (chl), "
              f"{int(prof.BBP700_INT.notna().sum())} (bbp)")

        integrals, clim_lons, clim_lats = load_climatology(box, poc_relation=args.poc)
        print(f"  climatology box: {len(clim_lats)} x {len(clim_lons)} cells")

        prof = match_climatology(prof, integrals, clim_lons, clim_lats,
                                 wrap=box["lon0"] > box["lon1"])
        lost_float = int(prof["CHLA_INT"].isna().sum())
        lost_clim = int((prof["CHLA_INT"].notna() & prof["CHLA_CLIM"].isna()).sum())
        print(f"  usable chl anomalies: {int(prof['CHLA_ANOM'].notna().sum())}/{len(prof)} "
              f"({lost_float} lost to float depth coverage, "
              f"{lost_clim} to climatology depth coverage)")
        prof.insert(0, "EVENT", event)

        during = (prof["JULD"] >= box["date_start"]) & (prof["JULD"] <= box["date_end"])
        prof["PERIOD"] = np.where(prof["JULD"] < box["date_start"], "before",
                          np.where(prof["JULD"] > box["date_end"], "after", "during"))
        for var in VARIABLES:
            col = f"{var}_ANOM"
            print(f"    {var:8s} anomaly  median {prof[col].median():+.4g} | "
                  + " | ".join(f"{p} {g.median():+.4g}"
                               for p, g in prof.groupby('PERIOD')[col]))

        out = os.path.join(ANOM_DIR, f"{event}_integrated_anomalies_{ZMAX:.0f}m.csv")
        prof.to_csv(out, index=False)
        print(f"  wrote {out}")
        print(f"  wrote {plot_anomaly_timeseries(prof, event, box, FIG_DIR)}")


if __name__ == "__main__":
    main()
