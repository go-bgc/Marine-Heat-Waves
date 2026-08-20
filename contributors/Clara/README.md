# BGC-Argo float processing and anomalies for marine heatwave boxes

Two scripts that take BGC-Argo profiles inside a marine heatwave box, quality
control and convert them, and express each profile as an anomaly against the
SOCA monthly climatology.

| script | what it does |
|---|---|
| `mhw_float_processing.py` | box selection → QC → despiking → 1 m regrid → BBP→POC/Cphyto → depth-time section plots |
| `mhw_float_anomalies.py`  | integrates each profile 0–`zmax` m, matches it to the SOCA climatology for its month and nearest cell, and plots the anomaly time series |

`mhw_float_anomalies.py` imports `mhw_float_processing.py`, so **keep both
files in the same folder**.

The method follows Vives et al. (2025, *Global Biogeochemical Cycles*,
10.1029/2024GB008327) §2.1–2.1.1 for QC, despiking and regridding, and Graff
et al. (2015) with Boss et al. (2013) for phytoplankton carbon. Every choice
is documented in the module docstrings — read those before changing anything.

## Install

```bash
pip install -r requirements.txt
```

## Data you need to supply

None of it ships with the repo; it is too large and, in the CMEMS case, needs
an account. Put it all under one folder — the "data dir".

```
<data-dir>/
├── MHW_list.csv                          # box definitions (in this repo)
├── floats_by_event/
│   ├── NA2023/*_Sprof.nc                 # BGC-Argo synthetic profiles
│   └── SWPac2023/*_Sprof.nc
└── SOCA_monthly_clim_chl_bbp_global.nc   # CMEMS climatology (anomalies only)
```

**`MHW_list.csv`** — columns `name,date_start,date_end,lon0,lon1,lat0,lat1`.
A box with `lon0 > lon1` crosses the dateline and is handled.

**Sprof files** (~1.4 GB for the two 2023 boxes) — free, no account, from the
Argo GDAC at <https://data-argo.ifremer.fr/dac/>. `fetch_mhw_floats.py` in this
repo will find and download the floats for each box.

**SOCA climatology** (~1 GB) — a monthly (month-of-year) climatology of
`chl` and `bbp` built from CMEMS `cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg`
(MULTIOBS_GLO_BIO_BGC_3D_REP_015_010). Needs a free Copernicus Marine account
and the `copernicusmarine` toolbox. Only `mhw_float_anomalies.py` uses it.

## Telling the scripts where the data is

Three ways, first hit wins:

```bash
python mhw_float_processing.py --data-dir /path/to/data     # 1. flag
export MHW_FLOAT_DATA=/path/to/data                         # 2. env var
                                                            # 3. the script's own folder
```

The climatology resolves the same way via `--clim-file` / `$MHW_SOCA_CLIM`,
falling back to `<data-dir>/SOCA_monthly_clim_chl_bbp_global.nc`.

## Run

```bash
python mhw_float_processing.py
python mhw_float_anomalies.py --zmax 100
```

Useful flags: `--events NA2023` (one box), `--poc cetinic` (swap the POC
relation), `--bbp-qc all` (keep every BBP flag rather than dropping 3 and 4),
`--max-float-figs 5` (cap the per-float figures), `--zmax` (integration depth).

## Two things to know before trusting the numbers

**The integration depth is limited by the climatology, not the floats.**
SOCA's valid depth shortens with depth: in the North Atlantic box 17% of cells
are empty throughout, but the NaN fraction climbs from 18% at 100 m to 54% at
200 m. Integrating to 200 m therefore discards more than half the box.
`--zmax 100` roughly doubles the usable sample; the script prints the coverage
either way.

**POC is indicative.** The Johnson et al. (2017) relation
(`31200 × bbp700 + 3.04`, µmol C kg⁻¹) is confirmed; the Cetinić et al. (2012)
alternative is not, and the two disagree by orders of magnitude at open-ocean
backscatter because Cetinić was fitted over bloom conditions. The script warns
at runtime. If POC matters, note that the CMEMS product carries POC as a
native variable — re-downloading the climatology with `poc` included removes
the conversion entirely.

## Outputs

```
<data-dir>/processed/<event>_profiles.parquet
<data-dir>/anomalies/<event>_integrated_anomalies_<zmax>m.csv
<data-dir>/figures/processed/<event>/<event>_<wmo>_sections.png
<data-dir>/figures/processed/<event>_all_floats_sections.png
<data-dir>/figures/anomalies_matchup/<event>_integrated_anomaly_timeseries_<zmax>m.png
```
