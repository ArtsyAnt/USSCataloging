# USSCataloging

NRAO Summer Project 2026 — cross-matching Fermi-LAT gamma-ray source ellipses (4FGL/FL16Y) against
multiwavelength radio/infrared catalogs (MALS, TGSS, NVSS, RACS, VLASS, SUMSS, SPICE-RACS, GLIMPSE) to find
candidate counterparts, with a focus on ultra-steep-spectrum (USS) and compact-steep-spectrum (CSS) radio
sources, and on how unassociated Fermi sources might be resolved by newer, deeper surveys.

See `CLAUDE.md` for the full architecture writeup (class design, per-script responsibilities, performance
notes, known gotchas). This file is the quick-start + current-state summary.

## Setup

```
pixi install
```

Dependencies: python 3.11, numpy, astropy, matplotlib, requests, astroquery, regions (see `pixi.toml`).

## Getting the catalogs

```
pixi run python code/catalog_download.py                                   # everything
pixi run python code/catalog_download.py --only RACS_low mals_all_bands    # just these
```

Downloads 4FGL (HEASARC, latest release auto-discovered) plus TGSS/NVSS/RACS_low/mals_all_bands/GLIMPSE
(VizieR) into `catalogs/`. Safe to re-run after a partial/interrupted run — it skips any VizieR catalog that
already has a matching FITS file on disk (`--no-skip-existing` to force a re-download). AllWISE is not
bulk-downloaded (~747M rows); see `download_wise_for_fermi_ellipses()` in that file for the per-source
cone-search alternative.

`catalogs/` currently on disk:

| File | Contents |
|---|---|
| `gll_psc_v22/v27/v31/v35.fit` | 4FGL DR1–DR4 |
| `gll_psc_v41.fit` | FL16Y (16-year), the current latest release |
| `TGSS_VIII_97_catalog.fits` / `_table1.fits` | TGSS ADR1 (VizieR VIII/97), 150 MHz |
| `NVSS_VIII_65_nvss.fits` | NVSS (VizieR VIII/65), 1.4 GHz |
| `mals_all_bands_*_catalog.fits` / `_tablea1.fits` / `_tablef1.fits` | MALS DR1 (VizieR J/ApJS/270/33), ~900-1600 MHz, one sub-table per spectral window |
| `RACS_low`, `GLIMPSE` | **not on disk** — see Current Status below |

## Sanity checks / tests

There's no test runner — validation is done via standalone scripts that independently recompute a result by
hand and diff against the pipeline's output:

```
pixi run python code/test_fermi_tgss_ellipse_match.py   # is_in_ellipse_V2 vs. hand-computed ellipse membership (TGSS + NVSS)
```

Last run (latest FL16Y release, TGSS + NVSS): 7060 associated Fermi sources; 870 have ≥1 TGSS candidate
(964 total candidates), 4334 have ≥1 NVSS candidate (9807 total candidates); spot-check against 25 (TGSS) +
40 (NVSS) hand-recomputed candidate pairs found 0 mismatches.

When adding a new sanity check, follow that pattern (recompute independently, report mismatches) rather than
introducing a test framework.

## The master output table

```
pixi run python code/build_fermi_master_table.py
```

Produces/updates two accumulating FITS tables in `catalogs/`, upserted (not duplicated) on re-run:

- **`fermi_matched_master.fits`** — one row per latest-release (FL16Y/v41) Fermi source, upserted by
  `Source_Name`. Columns:
  - all original FL16Y columns, plus `associated` (bool, from `CLASS1`)
  - **release history**: `CLASS1_<release>` / `SemiMajor_<release>` for v22/v27/v31/v35/v41 (position-matched
    to the v41 reference frame, 0.05 deg radius — same approach as `compare_fermi_releases.py`), and
    `ever_flipped_unassoc_to_assoc` (bool) — did this source appear unassociated in an earlier release and
    associated in a later one. This is the "how has this population changed over time" record.
  - **multiwavelength matches**, per radio survey `<name>` in `RADIO_CATALOGS` (currently TGSS, NVSS,
    mals_all_bands):
    `n_<name>_candidates`, `closest_<name>_ang_sep`, `closest_<name>_flux` (**mJy**, normalized from each
    survey's native flux unit - TGSS's `Sp` column is Jy, NVSS's `S1.4` and MALS's `Flux` are mJy, so this
    conversion matters and is done explicitly in `build_fermi_master_table.py`, not left as a raw column
    read). For already-**associated** sources this is the single closest counterpart (we already know the
    source is real, so one best match is enough); for **unassociated** sources the full candidate list lives
    in the companion table below instead, since the goal there is casting a wide net rather than picking a
    winner. Note: MALS has one row per (physical source, spectral window), so `n_mals_all_bands_candidates`
    and "closest by ang_sep" are per-row, not deduplicated per physical source.
  - **spectral index**, per pair of radio surveys that both produced a closest match for a source, column
    name alphabetical by survey name (e.g. `spectral_index_NVSS_TGSS`, `spectral_index_TGSS_mals_all_bands`),
    computed via the same `flux ~ freq^alpha` definition used in `assoc_catalog_script_test_2.calculate_spectral_index`.
    Frequency comes from `ref_freq_mhz` for single-frequency surveys (TGSS, NVSS) or the per-row `Freq`
    column for MALS (`freq_col`). Returns NaN when the two frequencies are within 20% of each other — alpha's
    denominator is `log(nu1/nu2)`, so near-equal frequencies blow noise up into physically meaningless alpha.
    Only computed where two data points exist.
  - **`adopted_spectral_index`, `adopted_spectral_index_source`** — a single "the" spectral index per source,
    picked in priority order from `ADOPTED_SPECTRAL_INDEX_PRIORITY`: MALS-TGSS first (deeper survey, flux
    scale directly comparable to TGSS in prior MALS work), falling back to NVSS-TGSS only when a source has
    no MALS candidate. Validated 2026-07-23: 72 sources adopted via MALS-TGSS, 785 via NVSS-TGSS fallback,
    857 total with an adopted index (mean alpha -1.03, median -0.71, 293 steep [alpha < -1], 230 USS
    [alpha < -1.4]). To change the priority order or add a third fallback pair, edit
    `ADOPTED_SPECTRAL_INDEX_PRIORITY` — it's a list of `(pair_name, column_name)` tried in order, first
    finite value wins.
- **`fermi_unassoc_candidates.fits`** — tidy companion table, one row per (unassociated Fermi source, survey,
  candidate): `Source_Name, survey, candidate_ra, candidate_dec, ang_sep, flux`. Fully regenerated each run
  (derived from whatever radio catalogs are currently loaded, not upserted).

See "Extending the pipeline" below for how to add a new radio survey.

## Shrunk-ellipse unassociated sub-table + plot

```
pixi run python code/build_shrunk_unassoc_subtable.py
pixi run python code/plot_shrunk_unassoc_ellipses.py
```

First cut at "can deeper surveys resolve unassociated sources": `build_shrunk_unassoc_subtable.py` takes
`fermi_matched_master.fits`, keeps only **unassociated** sources whose 95%-confidence ellipse **shrank**
between the earliest release they're matched in and v41/FL16Y, and writes
`catalogs/fermi_shrunk_unassoc_subtable.fits` (769 sources, latest run) — position/ellipse params, oldest vs.
newest release + shrink fraction, candidate counts, and an `overlap_group` id (union-find over
circle-radius-approximated ellipse overlap, i.e. pairs whose separation is within their summed semi-major
axes — a deliberately conservative stand-in for a true ellipse-ellipse intersection test, good enough to
decide whether two sources' candidates should be shown together). Currently 0 sources share a group (nearest
pair in this set is 0.31° apart vs. ~0.13° typical summed semi-major axes) — expected to change once
RACS_low/GLIMPSE land and add denser candidate coverage.

`plot_shrunk_unassoc_ellipses.py` renders `catalogs/fermi_shrunk_unassoc_ellipses.html` (Bokeh, standalone —
no server needed, just open it in a browser): a dropdown (one entry per source, sorted by shrink fraction)
swaps the plotted Fermi ellipse and its radio candidates (colored by survey, hover for ang_sep/flux) via
client-side `CustomJS` — no Python callback server required. Sources in the same `overlap_group` are drawn
together in one panel.

## Which script do I actually run?

There are two separate, **independent** pipelines in `code/` that both cross-match Fermi against radio
catalogs — they don't call each other and don't need to run in any particular order relative to each other:

- **`build_fermi_master_table.py`** — the accumulating, upserted master table (`fermi_matched_master.fits` +
  `fermi_unassoc_candidates.fits`). This is the one to run for the current MALS/adopted-spectral-index work;
  its config (`RADIO_CATALOGS`, `ADOPTED_SPECTRAL_INDEX_PRIORITY`) is at the top of the file and paths are
  relative (`../catalogs/`), so it runs as-is from `code/`.
- **`assoc_catalog_script_test_2.py`** — the original/actively-developed per-source association+spectral-index
  script, writes its own separate output (`<year>yr_grouped_ellipses.fits`) and has its own `radio_catalogs`
  config with the same shape. **Before running this one**, check `catalog_home_dir` near the top of `main()` —
  it's still hardcoded to a contributor's absolute path (`/Users/mario/Coding/...`) and needs updating to your
  own checkout's `catalogs/` path first, or it will fail to find any files.

If you're extending multiwavelength coverage (new survey, new spectral-index logic), you likely only need to
touch `build_fermi_master_table.py` — `assoc_catalog_script_test_2.py` is kept for its own separate
association-table output, not as an upstream dependency of the master table.

## Extending the pipeline

**To add a new radio survey** (once it's actually downloaded and its column schema checked — see Gotchas in
CLAUDE.md for why raw VizieR column names vary survey-to-survey):
1. Add an entry to `RADIO_CATALOGS` in `build_fermi_master_table.py`: glob pattern, RA/Dec/axis/flux column
   names, whether RA/Dec are sexagesimal, axis unit, and either `ref_freq_mhz` (fixed-frequency survey) or
   `freq_col` (per-row frequency, like MALS) — exactly one of the two.
2. Re-run `build_fermi_master_table.py`. Existing rows gain the new survey's `n_/closest_` columns, and
   `spectral_index_<A>_<B>` gets computed automatically against every other survey with a match for the same
   source.
3. If the new survey should factor into the single "adopted" number, add its pair to
   `ADOPTED_SPECTRAL_INDEX_PRIORITY` in priority order — first entry with a finite value wins per source.

**To change which spectral-index pair is preferred**, just reorder `ADOPTED_SPECTRAL_INDEX_PRIORITY` and
re-run; no other code needs to change.

## Current status / open threads

- **Only TGSS, NVSS, and mals_all_bands are actually on disk and wired into `RADIO_CATALOGS`.** RACS_low and
  GLIMPSE are not — despite earlier notes in this file implying they were mid-download, neither ever
  completed successfully (checked `catalogs/` and `logs/catalog_download.log` on 2026-07-23: no RACS_low or
  GLIMPSE FITS files exist, and the log shows only a RACS_low hard-timeout failure with no success line for
  either).
  - **RACS_low is a known, unresolved blocker**: the VizieR `row_limit=-1` pull hangs and hits
    `catalog_download.py`'s 900s hard timeout (see `hard_timeout()` and the gotcha in CLAUDE.md). Re-running
    with a longer timeout will not fix this — the real fix is paginating the query (VizieR's TAP/async
    interface, or a bounded `row_limit` with offset looping) rather than pulling all rows in one unlimited
    request.
  - **GLIMPSE** (Spitzer/IRAC mid-infrared Galactic-plane source catalog, VizieR `II/293`) has simply never
    been confirmed to complete — worth a fresh `catalog_download.py --only GLIMPSE` run and a check of
    `logs/catalog_download.log` for errors before assuming it's close.
  - Once either lands, its column schema needs checking (RA/Dec sexagesimal-or-not, axis column names/units,
    flux column, survey frequency) before adding it to `RADIO_CATALOGS` — see "Extending the pipeline" above.
- **Master table build**: `build_fermi_master_table.py` currently wired up for TGSS + NVSS + mals_all_bands;
  re-run after RACS_low/GLIMPSE are added to extend coverage and spectral-index pairs.
- **Planned: Bokeh plotting.** Not yet started. Candidate first plots, once the master table has more than
  two surveys in it:
  - CLASS1 vs. `ever_flipped_unassoc_to_assoc` vs. release, to visualize how the associated/unassociated split
    has shifted release to release (a Bokeh-interactive version of the release-history table above).
  - `Conf_95_SemiMajor` (ellipse size) vs. release, per source, to show confidence-region shrinkage over time
    (ties into `compare_fermi_releases.py`'s shrinkage-fraction stat).
  - Spectral index (`spectral_index_TGSS_NVSS`, etc.) distribution, split by `associated`, to look for
    USS/CSS-like candidates among the unassociated population — this is the "can deeper surveys resolve
    unassociated sources" question the multiwavelength-match columns are aimed at.
  - Sky positions (RA/Dec) of unassociated sources colored by `n_<survey>_candidates`, to see whether
    candidate-rich unassociated sources cluster anywhere (survey coverage gaps, Galactic plane confusion,
    etc.).
  - No implementation decisions made yet (layout, whether standalone HTML vs. a served Bokeh app, which
    columns get interactive tooltips) — flag before starting so this doesn't turn into unplanned scope.
