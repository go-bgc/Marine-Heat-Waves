"""
mhw_float_processing.py — build clean, converted BGC-Argo profiles for the two
2023 marine heatwave boxes, then plot them as depth-time sections.

FLOW:
  1. read the MHW boxes from MHW_list.csv (the two 2023 rows)
  2. read every Sprof in floats_by_event/<event>/ and keep the levels that
     fall inside the box and the time window
  3. LEVEL QC — ADJUSTED fields only, never a fall-back to the raw field:
       CHLA   : CHLA_ADJUSTED, keep QC flags 1, 2, 5, 8
       BBP700* : BBP700_ADJUSTED, --bbp-qc controls the flags 

     *BBP700 HAS TWO DEFENSIBLE ANSWERS AND THEY CONFLICT:
       --bbp-qc vives2025 (DEFAULT) drops flags 3 and 4, following Vives et al.
                      (2025, GBC) Sect. 2.1 after Bittig et al. (2019) and
                      Johnson et al. (2017)
       --bbp-qc all   keeps every flag, on the argument that flag-3 bbp levels
                      are mostly particle-cluster spikes rather than bad data
                      and the despiking filter is the right tool for those
     Which one is used is printed at runtime and stored in the output.

  4. PROFILE QC (Vives et al. 2025, Sect. 2.1) — a profile is dropped unless
       (a) its shallowest pressure is in the upper FIRST_PRES_MAX m, and
       (b) the upper 300 m carries at least MIN_OBS_300 observations of BOTH
           chl and bbp
     These remove profiles too sparse to integrate meaningfully, which a
     level-by-level filter cannot catch.

  5. DESPIKING (Su et al. 2021; Cornec, Claustre et al. 2021), on the
     vertical resolution:
       chl    : 7-point running median  (Vives et al. 2025 Sect. 2.1.1)
       bbp700 : 5-point running median then 5-point running mean, the bbp
                filter of Cornec et al. (2021); --bbp-smooth median7 swaps it
                for the same 7-point median used on chl
     Raw fields are kept as <VAR>_RAW so the filter can be changed.

  6. EXTRAPOLATION/intrapolation (Vives et al. 2025, Sect. 2.1) — each profile is extrapolated from
     its shallowest level (<10 m) up to the surface, then chl and bbp are
     interpolated to a 1 m grid from 0 to 300 m. Two extra smoothed fields are
     built ON that 1 m grid for later DCM/DBM identification, exactly as the
     paper separates them from the analysis fields:
       CHLA_DCM   : 5-point running median   (for locating true DCMs)
       BBP700_DBM : 5-point running median then 5-point running mean
  7. conversions from BBP700:
       -> POC     Johnson et al. (2017) SOCCOM relation (default), or
                  Cetinic et al. (2012) North Atlantic relation
       -> Cphyto  Graff et al. (2015) Eq. 1, after Boss et al. (2013) /
                  Boss and Haentjens (2016) Eq. 2 for the 700 -> 470 nm shift
  8. write one parquet per box, then plot sections: one figure per float,
     and one figure with every profile from every float in the box

NO CHL SLOPE CORRECTION** is applied/needed as long as the data have been downloaded after June 1, 2026.
**Vives et al. (2023) multiply adjusted chl
by 2 to undo the factory /2 and then divide by 3.79 for Southern Ocean iron
limitation. The x2 only applies to files processed under the older DAC
convention; every Sprof here has DATE_UPDATE 2026-06 or later, so the adjusted
chl already carries the corrected slope. The /3.79 is SO-specific and would
bias the North Atlantic box. The Uchida et al. (2019) sub-200 m chl mask is
also left out, by choice.

Usage:
    python mhw_float_processing.py                    # both boxes, all plots
    python mhw_float_processing.py --events NA2023    # one box
    python mhw_float_processing.py --max-float-figs 5 # only 5 per-float figures
    python mhw_float_processing.py --poc cetinic      # swap the POC relation
    python mhw_float_processing.py --bbp-qc all       # keep every BBP flag
                                                      # (default: vives2025)
    python mhw_float_processing.py --no-interp        # skip the 1 m regrid
    python mhw_float_processing.py --data-dir /path/to/data   # data elsewhere

Data the script expects under --data-dir (see README.md):
    MHW_list.csv
    floats_by_event/<event>/*_Sprof.nc
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cmocean import cm as cmo

# ------------
# Data root. Resolution order, first hit wins:
#   1. --data-dir on the command line
#   2. the MHW_FLOAT_DATA environment variable
#   3. the directory this script lives in
# So a checkout with the data alongside the scripts runs with no edits, and
# nobody has to hand-edit a path to use this on their own machine.
HERE = os.environ.get("MHW_FLOAT_DATA", os.path.dirname(os.path.abspath(__file__)))
MHW_LIST = FLOAT_DIR = OUT_DIR = FIG_DIR = None      # filled by set_data_dir()


def set_data_dir(path=None):
    """Point the module at a data root and rebuild the derived paths.

    Called from main(), and safe to call from a notebook or another script
    before using any of the loaders.
    """
    global HERE, MHW_LIST, FLOAT_DIR, OUT_DIR, FIG_DIR
    if path:
        HERE = os.path.abspath(os.path.expanduser(path))
    MHW_LIST = os.path.join(HERE, "MHW_list.csv")
    FLOAT_DIR = os.path.join(HERE, "floats_by_event")
    OUT_DIR = os.path.join(HERE, "processed")
    FIG_DIR = os.path.join(HERE, "figures", "processed")
    return HERE


set_data_dir()

EVENTS = {"NA2023": "NA 2023", "SWPac2023": "SWPac2023"}   # key -> row in MHW_list.csv

# ------------
CHLA_QC_GOOD = [b"1", b"2", b"5", b"8"]   # CHLA_ADJUSTED_QC values we keep
BBP_QC_MODES = {
    # Vives et al. (2025) GBC 10.1029/2024GB008327 Sect. 2.1, after
    # Bittig et al. (2019) and Johnson et al. (2017): drop flags 3 and 4
    "vives2025": [b"1", b"2", b"5", b"8"],
    "all": None,                          # keep every flag
}

DELTA_DAYS = 365       # pad either side of the MHW window, as in flag_mhw_natl.ipynb
DEPTH_MAX = 320.0      # levels read (analysis is 0-200 m; profile QC needs 300 m)

# despiking, Vives et al. (2025) Sect. 2.1.1 / Cornec et al. (2021)
CHL_MEDIAN_WINDOW = 7
BBP_MEDIAN_WINDOW = 5
BBP_MEAN_WINDOW = 5
DCM_MEDIAN_WINDOW = 5      # on the 1 m grid, for DCM/DBM identification

# profile-level QC, Vives et al. (2025) Sect. 2.1
FIRST_PRES_MAX = 10.0      # shallowest level must be this shallow
MIN_OBS_300 = 20           # observations required in the upper 300 m

# 1 m regrid, Vives et al. (2025) Sect. 2.1
INTERP_DZ = 1.0
INTERP_MAX = 300.0
INTERP_GRID = np.arange(0.0, INTERP_MAX + INTERP_DZ, INTERP_DZ)

# set from the CLI in main(); module-level so despike() can see it
BBP_SMOOTH_MODE = "cornec"
BBP_QC_MODE = "vives2025"

# BBP700 -> POC, JOhnson et al. (2017)
#JOHNSON_POC = 31200 * BBP700 + 3.04
# BBP700 -> Cphyto, Graff et al. (2015) Eq. 1 + Boss et al. (2013) Eq. 2
BBP470_FACTOR = (470.0 / 700.0) ** -0.78          # = 1.36440
GRAFF_SLOPE, GRAFF_INTERCEPT = 12128.0, 0.59      # mg C m-3

# BBP700 -> POC. Two published relations, neither universal:
#   johnson : Johnson et al. (2017), SOCCOM floats, umol C kg-1 -> converted here
#             to mg C m-3 with a nominal seawater density
#   cetinic : Cetinic et al. (2012), North Atlantic Bloom, already mg C m-3
# The johnson slope/intercept are confirmed (31200 * BBP700 + 3.04). The
# CETINIC coefficients are NOT: check them against the primary paper before
# using that mode. The two disagree violently at the backscatter these floats
# actually measure, because the SWPac2023 median bbp700 sits below the
# bloom-range backscatter Cetinic was fitted over and its negative intercept
# then dominates. poc_range_check() reports this at runtime.
#
# UNITS: Johnson is fitted in umol C kg-1. Both a native-unit column
# (POC_UMOL_KG) and a volumetric one (POC, mg C m-3) are written, so the raw
# relation stays visible and POC is still directly comparable with Cphyto.
#
# The cleaner fix if POC matters: the CMEMS/SOCA product these anomalies are
# referenced to (cmems_obs-mob_glo_bgc-chl-poc...) carries POC as a NATIVE
# variable. Re-downloading the climatology with `poc` included would make the
# comparison like-for-like and remove the conversion entirely.
POC_RELATIONS = {
    "johnson": dict(slope=3.12e4, intercept=3.04, units="umol/kg",
                    valid_bbp=(2e-4, 6e-3)),
    "cetinic": dict(slope=35422.0, intercept=-14.4, units="mg/m3",
                    valid_bbp=(1e-3, 1e-2)),
}
RHO_SW = 1025.0        # kg m-3, for the umol kg-1 -> mg m-3 conversion
MOLAR_C = 12.011       # g mol-1


# --------------------------------------------------------------------------- boxes
def load_mhw_boxes(events=None):
    """The 2023 rows of MHW_list.csv, keyed by short name.

    Returns {key: dict(lon0, lon1, lat0, lat1, date_start, date_end)}. A box
    with lon0 > lon1 crosses the dateline (SWPac2023 runs 150 E to 140 W); the
    notebook's `(lon > lon0) & (lon < lon1)` test returns zero rows for that
    case, so every longitude test here goes through in_box() instead.
    """
    mhw_dates = pd.read_csv(MHW_LIST)
    out = {}
    for key, row_name in EVENTS.items():
        if events and key not in events:
            continue
        sel = mhw_dates.loc[mhw_dates["name"] == row_name]
        if sel.empty:
            raise KeyError(f"{row_name!r} not found in {MHW_LIST}")
        lon0, lon1, lat0, lat1 = sel[["lon0", "lon1", "lat0", "lat1"]].values[0, :]
        start_time, end_time = sel[["date_start", "date_end"]].values[0, :]
        out[key] = dict(lon0=float(lon0), lon1=float(lon1),
                        lat0=float(lat0), lat1=float(lat1),
                        date_start=pd.Timestamp(start_time),
                        date_end=pd.Timestamp(end_time))
    return out


def in_box(lon, lat, box):
    """Inside the MHW box? Survives a box that crosses the dateline."""
    in_lat = (lat >= box["lat0"]) & (lat <= box["lat1"])
    if box["lon0"] <= box["lon1"]:
        in_lon = (lon >= box["lon0"]) & (lon <= box["lon1"])
    else:
        in_lon = (lon >= box["lon0"]) | (lon <= box["lon1"])
    return in_lat & in_lon


# --------------------------------------------------------------------------- QC
def qc_adjusted(ds, name, keep_flags):
    """One ADJUSTED field, NaN wherever its QC flag is not in keep_flags.

    keep_flags=None keeps every flag (used for BBP700). There is no fall-back
    to the raw field: for CHLA the raw values are flagged 3 across the board,
    and mixing an adjusted and an unadjusted field inside one profile would
    put a calibration step in the middle of the water column.
    """
    var = f"{name}_ADJUSTED"
    if var not in ds.variables:
        return None
    values = ds[var].values.astype("float64")
    if keep_flags is None:
        return values
    qc_var = f"{var}_QC"
    if qc_var not in ds.variables:
        return np.full_like(values, np.nan)
    flags = ds[qc_var].values
    flags = flags.astype("S1") if flags.dtype.kind in "SO" else flags
    return np.where(np.isin(flags, keep_flags), values, np.nan)


def despike(df):
    """Vertical despiking on the native profile resolution.

    chl gets a 7-point running median (Su et al. 2021, as used by Vives et al.
    2025 Sect. 2.1.1). bbp gets the Cornec, Claustre et al. (2021) filter — a
    5-point running median followed by a 5-point running mean — because bbp
    carries sharp particle-aggregate spikes that a median alone leaves a step
    at. Both are centred with min_periods=1, so the top and bottom of a profile
    survive instead of turning into NaN.

    Profiles are sorted by PRES first: a running filter over unsorted levels is
    meaningless, and Sprof level order is not guaranteed.
    """
    df = df.sort_values(["PLATFORM_NUMBER", "CYCLE_NUMBER", "PRES"]).copy()
    grouped = df.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"], sort=False)

    df["CHLA_RAW"] = df["CHLA"]
    df["BBP700_RAW"] = df["BBP700"]

    df["CHLA"] = grouped["CHLA"].transform(
        lambda s: s.rolling(CHL_MEDIAN_WINDOW, center=True, min_periods=1).median())

    if BBP_SMOOTH_MODE == "cornec":
        med = grouped["BBP700"].transform(
            lambda s: s.rolling(BBP_MEDIAN_WINDOW, center=True, min_periods=1).median())
        df["BBP700"] = med
        df["BBP700"] = df.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"], sort=False)["BBP700"].transform(
            lambda s: s.rolling(BBP_MEAN_WINDOW, center=True, min_periods=1).mean())
    else:
        df["BBP700"] = grouped["BBP700"].transform(
            lambda s: s.rolling(CHL_MEDIAN_WINDOW, center=True, min_periods=1).median())
    return df


def profile_coverage_qc(df):
    """Drop whole profiles that are too sparse to integrate (Vives et al. 2025).

    (a) the shallowest pressure must be in the upper FIRST_PRES_MAX m, and
    (b) the upper 300 m must carry at least MIN_OBS_300 observations of BOTH
        chl and bbp.
    Returns (kept, report) so the caller can print how many went and why.
    """
    keys = ["PLATFORM_NUMBER", "CYCLE_NUMBER"]
    upper = df[df["PRES"] <= INTERP_MAX]
    stats = upper.groupby(keys).agg(
        first_pres=("PRES", "min"),
        n_chl=("CHLA", "count"),
        n_bbp=("BBP700", "count"))

    shallow_ok = stats["first_pres"] <= FIRST_PRES_MAX
    count_ok = (stats["n_chl"] >= MIN_OBS_300) & (stats["n_bbp"] >= MIN_OBS_300)
    good = stats.index[shallow_ok & count_ok]

    report = dict(n_profiles=len(stats),
                  fail_first_pres=int((~shallow_ok).sum()),
                  fail_n_obs=int((~count_ok).sum()),
                  kept=len(good))
    kept = df.set_index(keys).loc[good].reset_index()
    return kept, report


def regrid_1m(df):
    """Extrapolate each profile to the surface, then interpolate chl and bbp to
    a 1 m grid from 0 to 300 m (Vives et al. 2025, Sect. 2.1).

    The surface extrapolation is constant, not linear: the shallowest good
    value is carried up to 0 m. Extrapolating a gradient through the top few
    metres of a fluorescence profile invents a quenching signal that is not in
    the data.

    Two extra fields are built ON this grid, kept separate from the analysis
    fields exactly as the paper separates them, for later DCM/DBM work:
      CHLA_DCM   5-point running median
      BBP700_DBM 5-point running median then 5-point running mean
    """
    keys = ["PLATFORM_NUMBER", "CYCLE_NUMBER"]
    meta_cols = ["JULD", "LATITUDE", "LONGITUDE"]
    out = []
    for (wmo, cycle), prof in df.groupby(keys, sort=False):
        prof = prof.sort_values("PRES")
        rec = {"PLATFORM_NUMBER": wmo, "CYCLE_NUMBER": cycle, "PRES": INTERP_GRID}
        for col in ("CHLA", "BBP700"):
            good = prof.dropna(subset=[col])
            if len(good) < 2:
                rec[col] = np.full(INTERP_GRID.shape, np.nan)
                continue
            z, v = good["PRES"].values, good[col].values
            # np.interp holds the end values flat outside the data range, which
            # is exactly the constant surface extrapolation we want
            interp = np.interp(INTERP_GRID, z, v)
            interp[INTERP_GRID > z[-1]] = np.nan      # never extrapolate downward
            rec[col] = interp
        block = pd.DataFrame(rec)
        for col in meta_cols:
            block[col] = prof[col].iloc[0]
        out.append(block)

    grid = pd.concat(out, ignore_index=True)
    grouped = grid.groupby(keys, sort=False)
    grid["CHLA_DCM"] = grouped["CHLA"].transform(
        lambda s: s.rolling(DCM_MEDIAN_WINDOW, center=True, min_periods=1).median())
    bbp_med = grouped["BBP700"].transform(
        lambda s: s.rolling(BBP_MEDIAN_WINDOW, center=True, min_periods=1).median())
    grid["BBP700_DBM"] = bbp_med
    grid["BBP700_DBM"] = grid.groupby(keys, sort=False)["BBP700_DBM"].transform(
        lambda s: s.rolling(BBP_MEAN_WINDOW, center=True, min_periods=1).mean())
    return grid


# --------------------------------------------------------------------------- conversions
def bbp_to_cphyto(bbp700):
    """Phytoplankton carbon, mg C m-3 (Graff et al. 2015 Eq. 1).

    bbp470 = bbp700 * (470/700)**-0.78 ; Cphyto = 12128*bbp470 + 0.59.
    Affine in bbp700, so Cphyto anomalies are a fixed 16547x the bbp anomaly.
    """
    return GRAFF_SLOPE * (bbp700 * BBP470_FACTOR) + GRAFF_INTERCEPT


def bbp_to_poc(bbp700, relation="johnson"):
    """Particulate organic carbon, mg C m-3.

    Johnson et al. (2017) is fitted in umol C kg-1 and converted here with a
    nominal density; Cetinic et al. (2012) is already volumetric. The two
    differ by roughly a factor of 1.5 at typical open-ocean bbp, so which one
    is used matters and is recorded in the output attributes.
    """
    if relation not in POC_RELATIONS:
        raise KeyError(f"POC relation must be one of {list(POC_RELATIONS)}")
    rel = POC_RELATIONS[relation]
    poc = rel["slope"] * bbp700 + rel["intercept"]
    if rel["units"] == "umol/kg":
        poc = poc * RHO_SW * MOLAR_C / 1000.0     # umol kg-1 -> mg m-3
    return poc


def poc_range_check(bbp700, relation):
    """Report how far the data sit outside a POC relation's fitted range, and
    how far the two relations disagree. Printed, not raised -- the point is to
    make the reader look, not to stop the run."""
    values = np.asarray(bbp700)
    values = values[np.isfinite(values)]
    if not values.size:
        return
    lo, hi = POC_RELATIONS[relation]["valid_bbp"]
    below = float((values < lo).mean())
    median = float(np.median(values))
    both = {k: float(bbp_to_poc(np.array([median]), k)[0]) for k in POC_RELATIONS}
    print(f"    POC relation '{relation}': {below:.0%} of levels below its "
          f"fitted bbp range ({lo:.0e}-{hi:.0e} m-1)")
    print(f"    at the median bbp of {median:.2e} m-1 the two relations give "
          + " vs ".join(f"{k} {v:.1f}" for k, v in both.items())
          + " mg C m-3 -- treat POC as indicative only")


def add_conversions(df, poc_relation="johnson"):
    """Cphyto, POC and the two ratios, level by level."""
    df = df.copy()
    df["CPHYTO"] = bbp_to_cphyto(df["BBP700"])
    rel = POC_RELATIONS[poc_relation]
    df["POC_NATIVE"] = rel["slope"] * df["BBP700"] + rel["intercept"]
    df["POC"] = bbp_to_poc(df["BBP700"], relation=poc_relation)
    # ratios formed depth by depth, before any averaging
    df["CHL_CPHYTO"] = df["CHLA"] / df["CPHYTO"].where(df["CPHYTO"] > 0)
    df["CHL_POC"] = df["CHLA"] / df["POC"].where(df["POC"] > 0)
    return df


# --------------------------------------------------------------------------- read
def read_sprof(path):
    """One Sprof file -> long DataFrame, one row per level per profile.

    Column names follow the Argo convention used across this project
    (PLATFORM_NUMBER, CYCLE_NUMBER, JULD, LATITUDE, LONGITUDE, PRES) so the
    output drops straight into MHW_funcs.get_oisst_flag and friends.
    """
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        chla = qc_adjusted(ds, "CHLA", CHLA_QC_GOOD)
        bbp = qc_adjusted(ds, "BBP700", BBP_QC_MODES[BBP_QC_MODE])
        if chla is None or bbp is None:
            return None

        n_prof, n_lev = ds.sizes["N_PROF"], ds.sizes["N_LEVELS"]
        pres = (ds["PRES_ADJUSTED"].values.astype("float64")
                if "PRES_ADJUSTED" in ds.variables else ds["PRES"].values.astype("float64"))
        if not np.isfinite(pres).any():
            pres = ds["PRES"].values.astype("float64")

        wmo = str(np.atleast_1d(ds["PLATFORM_NUMBER"].values)[0]).strip().strip("b'\" ")
        juld = pd.to_datetime(ds["JULD"].values)

        df = pd.DataFrame({
            "PLATFORM_NUMBER": np.repeat(wmo, n_prof * n_lev),
            "CYCLE_NUMBER": np.repeat(ds["CYCLE_NUMBER"].values, n_lev),
            "JULD": np.repeat(juld.values, n_lev),
            "LATITUDE": np.repeat(ds["LATITUDE"].values, n_lev),
            "LONGITUDE": np.repeat(ds["LONGITUDE"].values, n_lev),
            "PRES": pres.ravel(),
            "CHLA": chla.ravel(),
            "BBP700": bbp.ravel(),
        })
    finally:
        ds.close()

    df = df.dropna(subset=["PRES", "JULD", "LATITUDE", "LONGITUDE"])
    return df[df["PRES"] <= DEPTH_MAX]


def load_box_floats(event, box, poc_relation="johnson", interp=True):
    """Every float in one box: read, QC, despike, convert."""
    files = sorted(f for f in glob.glob(os.path.join(FLOAT_DIR, event, "*_Sprof.nc"))
                   if not os.path.basename(f).startswith("._"))
    if not files:
        raise FileNotFoundError(f"no Sprof files in {os.path.join(FLOAT_DIR, event)}")

    time0 = box["date_start"] - pd.Timedelta(days=DELTA_DAYS)
    time1 = box["date_end"] + pd.Timedelta(days=DELTA_DAYS)
    print(f"  {len(files)} Sprof files, window {time0.date()} to {time1.date()}", flush=True)

    keep = []
    for path in files:
        try:
            df = read_sprof(path)
        except Exception as exc:                      # one bad file must not stop the run
            print(f"    skipped {os.path.basename(path)}: {type(exc).__name__} {exc}")
            continue
        if df is None:
            print(f"    skipped {os.path.basename(path)}: no ADJUSTED CHLA/BBP700")
            continue
        df = df[(df["JULD"] >= time0) & (df["JULD"] <= time1)]
        df = df[in_box(df["LONGITUDE"].values, df["LATITUDE"].values, box)]
        df = df.dropna(subset=["CHLA", "BBP700"], how="all")
        if len(df):
            keep.append(df)

    df = pd.concat(keep, ignore_index=True)

    df, report = profile_coverage_qc(df)
    print(f"  profile QC: {report['kept']}/{report['n_profiles']} kept "
          f"({report['fail_first_pres']} with no level above {FIRST_PRES_MAX:.0f} m, "
          f"{report['fail_n_obs']} with <{MIN_OBS_300} obs in the upper 300 m)")

    df = despike(df)
    if interp:
        df = regrid_1m(df)
        print(f"  regridded to {INTERP_DZ:.0f} m, 0-{INTERP_MAX:.0f} m "
              f"({len(df):,} levels)")
    poc_range_check(df["BBP700"].values, poc_relation)
    df = add_conversions(df, poc_relation=poc_relation)
    df["MONTH"] = df["JULD"].dt.month
    df["YEAR"] = df["JULD"].dt.year
    return df


# --------------------------------------------------------------------------- plots
SECTION_VARS = [
    ("CHLA", "chl (mg m$^{-3}$)", cmo.algae),
    ("BBP700", "b$_{bp}$(700) (m$^{-1}$)", cmo.turbid),
    ("CPHYTO", "C$_{phyto}$ (mg C m$^{-3}$)", cmo.matter),
    ("POC", "POC (mg C m$^{-3}$)", cmo.matter),
    ("CHL_CPHYTO", "chl:C$_{phyto}$ (mg chl (mg C)$^{-1}$)", cmo.haline),
]


def _section(ax, grid, cmap, label, box, show_x):
    finite = grid.values[np.isfinite(grid.values)]
    if not finite.size:
        ax.text(0.5, 0.5, f"no {label}", transform=ax.transAxes, ha="center", va="center")
        ax.set_ylim(200, 0)
        return
    vmin, vmax = np.percentile(finite, [2, 98])
    pcm = ax.pcolormesh(grid.columns, grid.index.values, grid.values,
                        cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    for edge in (box["date_start"], box["date_end"]):
        ax.axvline(edge, color="k", lw=1.1)
    ax.set_ylim(200, 0)
    ax.set_ylabel("Pressure (dbar)", fontsize=9)
    ax.tick_params(labelsize=8, labelbottom=show_x)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    cbar = plt.colorbar(pcm, ax=ax, pad=0.015, fraction=0.03)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _grid(df, var, bins):
    """Profile x depth-bin grid, ready for pcolormesh. x is the profile time."""
    sub = df.dropna(subset=[var]).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["ZBIN"] = bins[np.clip(np.digitize(sub["PRES"].values,
                                           (bins[:-1] + bins[1:]) / 2.0),
                               0, len(bins) - 1)]
    return sub.pivot_table(index="ZBIN", columns="JULD", values=var, aggfunc="median")


def plot_float_sections(df, wmo, event, box, bins, out_dir):
    """One figure per float: every converted variable as a depth-time section."""
    sub = df[df["PLATFORM_NUMBER"] == wmo]
    fig, axes = plt.subplots(len(SECTION_VARS), 1, figsize=(13, 2.6 * len(SECTION_VARS)),
                             sharex=True)
    for i, (var, label, cmap) in enumerate(SECTION_VARS):
        _section(axes[i], _grid(sub, var, bins), cmap, label, box,
                 show_x=(i == len(SECTION_VARS) - 1))
    n_prof = sub[["PLATFORM_NUMBER", "CYCLE_NUMBER"]].drop_duplicates().shape[0]
    fig.text(0.005, 0.995,
             f"{event}   float {wmo}   {n_prof} profiles   "
             f"{sub.JULD.min():%Y-%m-%d} to {sub.JULD.max():%Y-%m-%d}",
             fontsize=12, fontweight="bold", va="top")
    fig.subplots_adjust(hspace=0.18)
    out = os.path.join(out_dir, f"{event}_{wmo}_sections.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_box_sections(df, event, box, bins, out_dir):
    """All profiles from all floats in the box, on one time axis.

    Profiles from different floats are interleaved by date, so this is a
    composite rather than a trajectory — read it for the seasonal cycle and
    the heatwave window, not for the fine structure of any one water mass.
    """
    fig, axes = plt.subplots(len(SECTION_VARS), 1, figsize=(14, 2.6 * len(SECTION_VARS)),
                             sharex=True)
    for i, (var, label, cmap) in enumerate(SECTION_VARS):
        _section(axes[i], _grid(df, var, bins), cmap, label, box,
                 show_x=(i == len(SECTION_VARS) - 1))
    n_prof = df[["PLATFORM_NUMBER", "CYCLE_NUMBER"]].drop_duplicates().shape[0]
    fig.text(0.005, 0.995,
             f"{event}   all floats   {df.PLATFORM_NUMBER.nunique()} floats, "
             f"{n_prof} profiles   {df.JULD.min():%Y-%m-%d} to {df.JULD.max():%Y-%m-%d}",
             fontsize=12, fontweight="bold", va="top")
    fig.subplots_adjust(hspace=0.18)
    out = os.path.join(out_dir, f"{event}_all_floats_sections.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", nargs="*", default=None, help="subset of %s" % list(EVENTS))
    parser.add_argument("--poc", default="johnson", choices=list(POC_RELATIONS))
    parser.add_argument("--max-float-figs", type=int, default=None,
                        help="cap the number of per-float figures (default: all)")
    parser.add_argument("--dz", type=float, default=5.0, help="section depth bin, m")
    parser.add_argument("--bbp-qc", default="vives2025", choices=list(BBP_QC_MODES),
                        help="'vives2025' drops BBP flags 3 and 4 (Vives et al. 2025 "
                             "Sect. 2.1, after Bittig et al. 2019 / Johnson et al. 2017); "
                             "'all' keeps every flag")
    parser.add_argument("--bbp-smooth", default="cornec", choices=["cornec", "median7"],
                        help="'cornec' = 5-pt median then 5-pt mean; "
                             "'median7' = same 7-pt median used on chl")
    parser.add_argument("--no-interp", action="store_true",
                        help="skip the 1 m regrid and keep native levels")
    parser.add_argument("--data-dir", default=None,
                        help="folder holding MHW_list.csv and floats_by_event/ "
                             "(default: $MHW_FLOAT_DATA, else this script's folder)")
    args = parser.parse_args()

    set_data_dir(args.data_dir)
    if not os.path.exists(MHW_LIST):
        raise SystemExit(
            f"MHW_list.csv not found at {MHW_LIST}\n"
            f"Point --data-dir (or $MHW_FLOAT_DATA) at the folder that holds it. "
            f"See README.md for what the data folder must contain.")

    global BBP_QC_MODE, BBP_SMOOTH_MODE
    BBP_QC_MODE, BBP_SMOOTH_MODE = args.bbp_qc, args.bbp_smooth
    print(f"BBP700 QC: {args.bbp_qc} | BBP700 smoothing: {args.bbp_smooth} | "
          f"chl despike: {CHL_MEDIAN_WINDOW}-point running median")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    bins = np.arange(0.0, 200.0 + args.dz, args.dz)

    for event, box in load_mhw_boxes(args.events).items():
        wrap = "  (crosses the dateline)" if box["lon0"] > box["lon1"] else ""
        print(f"\n=== {event}  lat {box['lat0']} to {box['lat1']}, "
              f"lon {box['lon0']} to {box['lon1']}{wrap}", flush=True)

        df = load_box_floats(event, box, poc_relation=args.poc,
                             interp=not args.no_interp)
        n_prof = df[["PLATFORM_NUMBER", "CYCLE_NUMBER"]].drop_duplicates().shape[0]
        print(f"  {len(df):,} levels | {n_prof} profiles | "
              f"{df.PLATFORM_NUMBER.nunique()} floats")
        for var in ("CHLA", "BBP700", "CPHYTO", "POC"):
            print(f"    {var:11s} n={df[var].notna().sum():>7,}  "
                  f"median={df[var].median():.4g}")

        out = os.path.join(OUT_DIR, f"{event}_profiles.parquet")
        df.to_parquet(out, index=False)
        print(f"  wrote {out}")

        wmos = sorted(df["PLATFORM_NUMBER"].unique())
        if args.max_float_figs:
            counts = df.groupby("PLATFORM_NUMBER")["CYCLE_NUMBER"].nunique()
            wmos = list(counts.sort_values(ascending=False).head(args.max_float_figs).index)
        per_float_dir = os.path.join(FIG_DIR, event)
        os.makedirs(per_float_dir, exist_ok=True)
        for wmo in wmos:
            plot_float_sections(df, wmo, event, box, bins, per_float_dir)
        print(f"  wrote {len(wmos)} per-float section figures to {per_float_dir}")
        print(f"  wrote {plot_box_sections(df, event, box, bins, FIG_DIR)}")


if __name__ == "__main__":
    main()
