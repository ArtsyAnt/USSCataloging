# import itertools
from pprint import pprint
import glob
from copy import copy
import re
import os

from multiwavecataloging import Catalog
from multiwavecataloging import CatalogOverlayer
from multiwavecataloging import fermi_plot
from multiwavecataloging import spectral_window_merging_V2
from multiwavecataloging import global_index
from multiwavecataloging import calc_reduced_chi_square

from astropy import units as u
from astropy.coordinates import SkyCoord

from astropy.io import fits
from astropy.table import vstack
# too hard to try and find all the quantities right now but need to fix later!
from astropy.table import QTable, Table, Column, MaskedColumn

from astropy.modeling import models, fitting
import warnings
from astropy.utils.exceptions import AstropyWarning
from astropy.units import UnitsWarning

# Access astronomical databases
# from astroquery.vizier import Vizier

# For plots
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as ellipse
import time

# Data handling
import numpy as np
#import pymc as pm
#import arviz as az

# would replace in the future with adding it into the direct script!
def reduce_wise_to_radio_psf(
    radio_rows,
    wise_rows,
    psf_multiplier=1.0,
    max_radius_arcsec=None
):
    """
    Keep only WISE sources that fall within the positional/PSF
    region of at least one radio candidate.

    radio_rows and wise_rows are the per-row dictionaries already
    created by process_single_fermi().

    The radio `_maj_key` value is assumed to be in degrees,
    matching the existing code in process_single_fermi().
    """

    if not radio_rows or not wise_rows:
        return wise_rows

    # ------------------------------------------------------------
    # Radio coordinates
    # ------------------------------------------------------------

    radio_ra = np.asarray(
        [r[r['_ra_key']] for r in radio_rows],
        dtype=np.float64
    )

    radio_dec = np.asarray(
        [r[r['_dec_key']] for r in radio_rows],
        dtype=np.float64
    )

    radio_psf_deg = np.asarray(
        [r[r['_maj_key']] for r in radio_rows],
        dtype=np.float64
    )

    # Remove bad radio coordinates / PSFs
    valid_radio = (
        np.isfinite(radio_ra) &
        np.isfinite(radio_dec) &
        np.isfinite(radio_psf_deg) &
        (radio_psf_deg > 0)
    )

    if not np.any(valid_radio):
        return []

    radio_ra = radio_ra[valid_radio]
    radio_dec = radio_dec[valid_radio]
    radio_psf_deg = radio_psf_deg[valid_radio]

    # ------------------------------------------------------------
    # WISE coordinates
    # ------------------------------------------------------------

    wise_ra = np.asarray(
        [r[r['_ra_key']] for r in wise_rows],
        dtype=np.float64
    )

    wise_dec = np.asarray(
        [r[r['_dec_key']] for r in wise_rows],
        dtype=np.float64
    )

    valid_wise = (
        np.isfinite(wise_ra) &
        np.isfinite(wise_dec)
    )

    if not np.any(valid_wise):
        return []

    wise_valid_indices = np.flatnonzero(valid_wise)

    wise_ra_valid = wise_ra[valid_wise]
    wise_dec_valid = wise_dec[valid_wise]

    # ------------------------------------------------------------
    # One vectorized spatial search
    # ------------------------------------------------------------

    radio_coords = SkyCoord(
        ra=radio_ra * u.deg,
        dec=radio_dec * u.deg
    )

    wise_coords = SkyCoord(
        ra=wise_ra_valid * u.deg,
        dec=wise_dec_valid * u.deg
    )

    # search_around_sky requires one maximum radius.
    # We therefore use the largest radio PSF as the coarse search
    # radius and then apply the individual PSF below.
    search_radius = np.max(radio_psf_deg) * psf_multiplier

    if max_radius_arcsec is not None:
        search_radius = min(
            search_radius,
            max_radius_arcsec / 3600.0
        )

    radio_idx, wise_idx, separation, _ = (
        radio_coords.search_around_sky(
            wise_coords,
            search_radius * u.deg
        )
    )

    if len(wise_idx) == 0:
        return []

    # ------------------------------------------------------------
    # Exact per-radio PSF filtering
    # ------------------------------------------------------------

    # radio_idx tells us which radio source produced the pair.
    # wise_idx tells us which WISE source produced the pair.

    allowed_radius_deg = (
        radio_psf_deg[radio_idx] * psf_multiplier
    )

    keep_pairs = (
        separation.to_value(u.deg) <= allowed_radius_deg
    )

    matched_wise_indices = np.unique(
        wise_idx[keep_pairs]
    )

    if len(matched_wise_indices) == 0:
        return []

    # Convert back to indices in the ORIGINAL wise_rows list.
    original_wise_indices = wise_valid_indices[
        matched_wise_indices
    ]

    return [
        wise_rows[i]
        for i in original_wise_indices
    ]
    
def calculate_spectral_index(f1, f2, nu1, nu2):
    """Calculates alpha where Flux proportional to freq^alpha."""
    if f1 <= 0 or f2 <= 0 or nu1 == nu2 or nu1 <= 0 or nu2 <= 0:
        return np.nan
    return np.log(f1 / f2) / np.log(nu1 / nu2)

    
# this is for matching my radio catalogs
def load_and_prep_catalogs(catalog_configs, base_dir, cat_idx=None):
    catalog_objects = {}
    for config in catalog_configs:
        name = config['name']
        cols = config['cols']
        type2 = config['type']
        drop_cols = config.get('drop_cols', [])  
        run_flux_ratio = config.get('flux_ratio', True)
        #print(run_flux_ratio)
        
        # Perform the glob search using the base file name
        search_pattern = f'{base_dir}{name}*.fits'
        file_matches = glob.glob(search_pattern)

        if name =='WISE':
            file_matches.sort(reverse=False)
            file_matches = [file_matches[cat_idx]]
        
        if not file_matches:
            print(f'No files found matching pattern: {search_pattern}')
            continue
        
        # Catalog Class loop
        for file_path in file_matches:
            print(file_path)
            try:
                table = QTable.read(file_path, format='fits')
                #HARDCODED - RENAME JUST FOR NOW!
                if name == 'WISE':
                    print(table.colnames)
                    try:
                        for col in table.colnames:
                            if col not in ['wise_ang_sep', 'catalog_names']:
                                original_name = col.split('_', 1)[1]
                                table.rename_column(col, original_name)
                    except Exception as e:
                        print(e)

                    table['ra'] = table['ra'] *u.deg
                    table['dec'] = table['dec'] *u.deg
                    table = table[~table['designation'].mask]
                                        
                for col in drop_cols:
                    if col in table.colnames:
                        table.remove_column(col)
                catalog_instance = Catalog(table, cols, catalog_type=type2)
                if run_flux_ratio:
                    catalog_instance.flux_ratio()
                
                catalog_objects[name] = catalog_instance
                print(f'Successfully processed catalog wrapper for: {name}')
            except Exception as e:
                print(f'Error executing catalog preparation on {file_path}: {e}')
                
    return catalog_objects

def process_single_fermi(ellipse_index, fermi_catalog_sliced, global_dict_indexes, updated_comparision_catalogs, search_term_z, search_term_flux, grouped_table_value=False):
    # the single fermi source as a plain dict (no Table wrapper needed for a single row lookup)
    #print('check')
    fermi_source_dict = {col: fermi_catalog_sliced[col][ellipse_index] for col in fermi_catalog_sliced.colnames}
    fallback_dict = dict(fermi_source_dict)
    fallback_dict['closest_spx_idx'] = np.nan

    # all other catalogs
    values = [global_dict.get(ellipse_index, None) for global_dict in global_dict_indexes]

    # Step A: Collect ALL viable sources inside the semi-major axis ellipse.
    # updated_comparision_catalogs entries are already plain dicts of numpy arrays
    # (built in main()), so this stays numpy-indexed the whole way through instead
    # of round-tripping through astropy Table construction per catalog per source.
    all_candidate_rows = []
    all_candidate_rows_optical_infared = []
    for catalog_dict, val in zip(updated_comparision_catalogs, values):
        if val is None:
            continue
        mask = np.isin(catalog_dict['Indexes'], val)
        if not np.any(mask):
            continue

        colnames = list(catalog_dict.keys())
        cat_name = str(catalog_dict['catalog_names'][mask][0]) if 'catalog_names' in colnames else "survey"

        ra_col = next((c for c in colnames if 'RA' in c.upper()), 'RAJ2000')
        dec_col = next((c for c in colnames if 'DEC' in c.upper()), 'DEJ2000')
        # no any search! needs to be a strict search so theres no mistakes 
        maj_col = next((c for c in colnames if c in search_term_z), None)
        flux_col = next((c for c in colnames if c in search_term_flux), None)

        # print(maj_col)
        # print(flux_col)
        # print(cat_name)
        # taking all of the wise sources out by accident
        if not maj_col or not flux_col:
            continue


        # No re-check against ang_sep here: `mask` already selects exactly the
        # rows CatalogOverlayer.is_in_ellipse_V2 confirmed are inside the Fermi
        # source's own rotated semi-major/semi-minor ellipse (via global_dict_indexes).
        # A prior version additionally filtered ang_sep <= this radio catalog's own
        # beam major axis (maj_col) - a mismatched-scale check (Fermi confidence
        # ellipse vs. arcsec-scale radio beam) that dropped valid overlaps.
        n_valid = int(np.count_nonzero(mask))
        # slice every column down to this catalog's matched rows in one shot
        sub = {col: np.asarray(arr)[mask] for col, arr in catalog_dict.items()}
        # Clean and type-cast valid rows to floating-point numbers immediately
        sub[ra_col] = sub[ra_col].astype(float)
        sub[dec_col] = sub[dec_col].astype(float)
        sub['radio_flux_name'] = np.array([flux_col] * n_valid, dtype='U20')
        

        if 'spectral_index' not in sub:
            sub['spectral_index'] = np.full(n_valid, np.nan)
        else:
            spx = sub['spectral_index']
            if spx.dtype.kind in {'S', 'U', 'O'}:
                clean_idx = spx.astype(str)
                float_idx = np.empty(n_valid, dtype=np.float64)
                for idx, val_str in enumerate(clean_idx):
                    val_strip = val_str.strip()
                    if val_strip in ('', '" "', '--', 'nan', 'None'):
                        float_idx[idx] = np.nan
                    else:
                        try:
                            float_idx[idx] = float(val_strip)
                        except ValueError:
                            float_idx[idx] = np.nan
                sub['spectral_index'] = float_idx
            else:
                sub['spectral_index'] = spx.astype(np.float64)

        # Unpack rows to plain per-row dicts, carrying metadata as sidecar keys
        for idx in range(n_valid):
            row = {col: arr[idx] for col, arr in sub.items()}
            row['_catalog_base_name'] = cat_name
            row['_ra_key'] = ra_col
            row['_dec_key'] = dec_col
            row['_maj_key'] = maj_col
            row['_flux_key'] = flux_col
            if cat_name == 'WISE':
                all_candidate_rows_optical_infared.append(row)
            else:
                all_candidate_rows.append(row)

    if not all_candidate_rows and not all_candidate_rows_optical_infared:
        return fallback_dict

    
    if all_candidate_rows and all_candidate_rows_optical_infared:

        wise_before = len(all_candidate_rows_optical_infared)

        all_candidate_rows_optical_infared = (
            reduce_wise_to_radio_psf(
                all_candidate_rows,
                all_candidate_rows_optical_infared,
                psf_multiplier=1.0))

        wise_after = len(all_candidate_rows_optical_infared)

        print(
            f"WISE PSF reduction: "
            f"{wise_before:,} -> {wise_after:,}"
        )
    # hope this works for wise 
    # print('check Step B')
    # Step B: check for nearby sources across catalogs and propagate/derive spectral index.
    # One vectorized SkyCoord + broadcasted separation matrix replaces building a
    # scalar SkyCoord per candidate and calling .separation() inside the O(k^2) loop.
    # only focuses on radio
    num_sources = len(all_candidate_rows)
    #print(num_sources)
    ra_arr  = np.array([r[r['_ra_key']] for r in all_candidate_rows], dtype=float)
    dec_arr = np.array([r[r['_dec_key']] for r in all_candidate_rows], dtype=float)
    maj_arr = np.array([r[r['_maj_key']] for r in all_candidate_rows], dtype=float)
    has_freq = np.array(['ref_freq' in r for r in all_candidate_rows])
    freq_arr = np.array([r.get('ref_freq', np.nan) for r in all_candidate_rows], dtype=float)

    coords = SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg)
    sep_matrix = coords[:, None].separation(coords[None, :])

    # iteration order matches the original nested loop (idx_i ascending, idx_j > idx_i)
    # so in-place spectral_index mutations propagate identically pair-to-pair
    #only accepts alpha that is > -10 and is not np.nan/0 for the mals matching!
    
    for idx_i in range(num_sources):
        
        if not has_freq[idx_i]:
            continue
        row_i = all_candidate_rows[idx_i]
        for idx_j in range(idx_i + 1, num_sources):
            if not has_freq[idx_j]:
                continue
            nu1, nu2 = freq_arr[idx_i], freq_arr[idx_j]
            if nu1 == nu2:
                continue

            sep_ij = sep_matrix[idx_i, idx_j]
            limit_i = maj_arr[idx_i] * u.deg
            limit_j = maj_arr[idx_j] * u.deg
            if sep_ij <= limit_i or sep_ij <= limit_j:
                row_j = all_candidate_rows[idx_j]
                idx_i_val = row_i['spectral_index']
                idx_j_val = row_j['spectral_index']

                if not np.isnan(idx_i_val) and np.isnan(idx_j_val):
                    row_j['spectral_index'] = idx_i_val
                elif not np.isnan(idx_j_val) and np.isnan(idx_i_val):
                    row_i['spectral_index'] = idx_j_val
                elif np.isnan(idx_i_val) and np.isnan(idx_j_val):
                    f1 = float(row_i[row_i['_flux_key']])
                    f2 = float(row_j[row_j['_flux_key']])
                    calculated_alpha = calculate_spectral_index(f1, f2, nu1, nu2)
                    # print(calculated_alpha)
                    # print('test')

                    # DEBUG: see if the cal alpha limit is what is blocking tgss and racs!
                    # if calculated_alpha <= -10:
                    #     continue
                    # else:
                    row_i['spectral_index'] = calculated_alpha
                    row_j['spectral_index'] = calculated_alpha
                        # print('check')

    # have the wise reduction happen here since this will be our smallest amount of sources!

    if grouped_table_value == True:
        catalog_groups = {}
        for row in all_candidate_rows:
            # CHANGE change this in the unassociated version!
            try:
                catalog_groups.setdefault(row['_catalog_base_name'], []).append(row)
            except:
                pass
        for row in all_candidate_rows_optical_infared:
            try:
                catalog_groups.setdefault(row['_catalog_base_name'], []).append(row)
            except:
                pass

        # 1. Collect all rows that need to be merged into this Fermi group
        grouped_dict = []
        grouped_dict.append(dict(fermi_source_dict))
        # append the fermi source first!
        
        # then append the rest of the valid sources in it 
        for cat_base, rows in catalog_groups.items():
            if cat_base == 'WISE':
                # print(rows)
                grouped_dict.append(rows)
            # cat in a list for a empty valid_spx_rows_skip it
            valid_spx_rows = [r for r in rows if not np.isnan(r['spectral_index'])]
            if not valid_spx_rows:
                continue

            grouped_dict.append(valid_spx_rows)


        return grouped_dict
    #TO ADD - PUTS IN THE CLOSEST SOURCE WITH A FLAG, MERGES THE SPXIDX SOURCES 
    #return this two dictionaries and make into a table in end 
    #take the fermi name and append it for the object
    #if it is a grouped table add all of the rows to the flat_dict
    #need 
            
    else:
        # print('check Step C- not grouped')

        try:
            # Step C: Isolate the closest source per catalog that HAS a spectral index
            catalog_groups = {}
            for row in all_candidate_rows:
                # CHANGE change this in the unassociated version!
                catalog_groups.setdefault(row['_catalog_base_name'], []).append(row)
    
            flat_dict = dict(fermi_source_dict)
            global_closest_row = None
            global_min_sep = None
    
            #combining the rows only iterates once?
            for cat_base, rows in catalog_groups.items():
                valid_spx_rows = [r for r in rows if not np.isnan(r['spectral_index'])]
                
                if not valid_spx_rows:
                    continue
    
                closest_row = min(valid_spx_rows, key=lambda r: r['ang_sep'])
                prefix = f"{cat_base}_src1"
                for col, val in closest_row.items():
                    if col.startswith('_'):
                        continue
                        
                    flat_dict[f'{prefix}_{col}'] = val
    
                current_sep = closest_row['ang_sep']
                if global_min_sep is None or current_sep < global_min_sep:
                    global_min_sep = current_sep
                    global_closest_row = closest_row
    
            # Step D: attach the single globally-closest matched source and enforce dtype safety
            #print('check Step D')
            if global_closest_row is not None:
                # this is what causes the repeat columns ofr lcosest when i dont want that!
                # for col, val in global_closest_row.items():
                #     if col.startswith('_'):
                #         continue
                #     flat_dict[f'closest_{col}'] = val
                
                flat_dict['closest_spx_idx'] = global_closest_row['spectral_index']
                flat_dict['closest_catalog_name'] = global_closest_row['catalog_names']
                flat_dict['closest_ang_sep'] = global_closest_row['ang_sep']
                flat_dict['closest_flux_ratio'] = global_closest_row['FluxRatio']

            else:
                flat_dict['closest_spx_idx'] = np.nan
    
            for col in list(flat_dict.keys()):
                val = flat_dict[col]
                if hasattr(val, 'filled'):
                    val = val.filled(np.nan)
    
                col_upper = col.upper()

            return flat_dict
        except Exception:
            return fallback_dict



# instead of looping it I would want to have all the values computed at once with the respective nan values that I iniate, this would make this process much quicker!
def spec_fitter_mc(catalog):
    az.style.use("arviz-variat")

    # get the flx, err, and freq for all the spectral windows of all sources
    colnames = catalog.colnames
    freqs_names = [freq for freq in colnames if 'ref_freq_LSPW' in freq]
    flux_names  = [flux for flux in colnames if 'total_flux_LSPW' in flux]
    flux_err_names = [err for err in colnames if 'total_flux_e_LSPW' in err]
    
    observed_freqs = catalog[freqs_names].as_array().tolist()
    observed_fluxs = catalog[flux_names].as_array().tolist()
    observed_flux_errs = catalog[flux_err_names].as_array().tolist()
        
    log_freqs = np.log(np.array(observed_freqs))
    mean_fluxes = np.nanmean(np.array(observed_fluxs), axis=1)
    log_fluxes = np.log(np.array(observed_fluxs))
    log_flux_errors = np.array(observed_flux_errs)/np.array(observed_fluxs)
    
    log_freqs = np.nan_to_num(log_freqs, posinf=np.nan, neginf=np.nan)
    log_mean_fluxes = np.log(mean_fluxes)
    log_fluxes = np.nan_to_num(log_fluxes, posinf=np.nan, neginf=np.nan)
    log_flux_errors = np.nan_to_num(log_flux_errors, posinf=np.nan, neginf=np.nan)

    # loop through the source here!
    for iteration in range(len(catalog)):
        
        # get all the values for the different arrays
        c_log_value =log_mean_fluxes[iteration]
        log_freq = log_freqs[iteration]
        log_flux = log_fluxes[iteration]
        log_flux_error = log_flux_errors[iteration]
    
    # Create coordinate mapping for log_freq so ArviZ can recognize it as an axis coordinate
        coords = {"obs_dim": np.arange(len(log_freq)), "log_freq_coord": log_freq}
    # make sure the data is masked!
    # we are observing in log-log space so making the powerlaw equation linear
    # might be easier to just do one prior of just alpha and whatever the spectral 
    # delta method approximation (dont know why but it is the log err approx)
        valid_mask = ~np.isnan(log_freq) & ~np.isnan(log_flux) & ~np.isnan(log_flux_error)
        log_freq = log_freq[valid_mask]
        log_flux = log_flux[valid_mask]
        log_flux_error = log_flux_error[valid_mask]
        # try:
        
        with pm.Model() as spectral_model:            
            pm.Data('log_freq', log_freq, dims='obs_dim')
            # call one spectral index prior or I could call mutiple and seperate them based on spec index
            alpha  = pm.Normal('alpha', mu=-1, sigma=1.5)
            # should this be the deterministic value? mJy prior
            c = pm.Normal('c', c_log_value, sigma=2)
            
            # the expected mean value for the log space
            # since it is not constant it is only acceped under this changed module
            spec_flux_density = pm.Deterministic(r'$S_{\nu}$', c + alpha * log_freq, dims='obs_dim') 
            
            # liklihood function (for the bayesian analysis, normal models residulas)
            # check if this is actually accurate!
            obs_spec_flux_density = pm.Normal(r'Observed $S_{\nu}$', mu=spec_flux_density, sigma=log_flux_error, observed=log_flux, dim='obs_dim')
            # sampler
            # print(spectral_model.debug())
            sample = pm.sample(draws=1000, tune=1000)
        summary=pm.summary(sample)
        print(summary)

        # try this?
        with spectral_model:
            pm.sample_posterior_predictive(sample, var_names=[r'Observed $S_{\nu}$', r'$S_{\nu}$'], extend_inferencedata=True)
            
            # add the mu alpha
            dt = az.convert_to_datatree(sample)
            az.plot_trace_dist(sample, combined=True)
            
            az.plot_lm(x='log_freq', y=r'$S_{\nu}$', y_obs=r'Observed $S_{\nu}$',dt=dt, plot_dim='obs_dim')
            plt.show()
            
        # if float(summary.iloc[0,0])<=-1:
        #     # try this?
        #     with spectral_model:
        #         pm.sample_posterior_predictive(sample, var_names=[r'Observed $S_{\nu}$', r'$S_{\nu}$'], extend_inferencedata=True)
                
        #         # add the mu alpha
        #         dt = az.convert_to_datatree(sample)
        #         az.plot_trace_dist(sample, combined=True)
                
        #         az.plot_lm(x='log_freq', y=r'$S_{\nu}$', y_obs=r'Observed $S_{\nu}$',dt=dt, plot_dim='obs_dim')
        #         plt.show()
        # except:
        #     print(f'{len(log_flux)} non-nan spws?')

def main(mals_merged = True, associated=False, file_marker='unassociated', grouped_table_value=False):
    warnings.filterwarnings('ignore', category=AstropyWarning)
    warnings.simplefilter('ignore', category=UnitsWarning)
    # catalog call
    # define home dir
    '''CATALOG CALLS'''

    
    current_working_dir = os.getcwd()
    catalog_home_dir = current_working_dir + '/catalogs/'
    catalog_reduced_dir = current_working_dir + '/reduced_catalogs/'
    catalog_mals_all = current_working_dir + '/catalogs/mals_all_bands.fits'
    
    #catalog_home_dir = '/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/'
    #catalog_reduced_dir = '/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/reduced_catalogs/'
    #catalog_mals_all = '/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/mals_all_bands.fits'
    
    '''FIX: remember this decides the catalog names in the end I dont like how this is hardcoded find another method!!'''
    
    catalog_names = ['MALS_L', 'SPICE-RACS', 'TGSS', 'RACS_low','RACS_low', 'NVSS', 'VLASS', 'SUMSS', 'FIRST', 'WISE']
    # catalog_names = ['MALS_L']
    '''FERMI CATALOG CREATION'''
    # not spectral combined
    mals_all_sources= QTable.read(catalog_mals_all, format='fits')
    'Fermi Catalogs'
    # fermi_lat_surveys = glob.glob(catalog_home_dir + 'gll**.fit')
    # fermi_lat_surveys.sort(reverse=False)
    # folder_surveys = ['8_yr/', '10_yr/', '12_yr/', '14_yr/', '16_yr/']
    # catalog_years = [8, 10, 12, 14, 16]

    # 'TEMP REMOVE ONLY FOR 8 YEAR'
    # fermi_lat_surveys = fermi_lat_surveys[1:]
    # print(fermi_lat_surveys)
    # catalog_years = [10, 12, 14, 16]
    # folder_surveys = ['10_yr/', '12_yr/', '14_yr/', '16_yr/']
    

    'TEMP ONLY DO 14 YEAR'
    fermi_lat_surveys = glob.glob(catalog_home_dir+ 'gll**35.fit')
    folder_surveys = ['14_yr/']
    catalog_years = [14]
    print(fermi_lat_surveys)


    radio_catalogs = [
            {'name': 'mals_all_bands',
            'cols': ['RAJ2000', 'DEJ2000', 'maj_restoring_beam', 'min_restoring_beam', 'pa_restoring_beam', 
                    'total_flux', 'peak_flux', 'spectral_index_spwfit', 'spectral_index_spwfit_e', 's_code', 'spw_id/WB'],
            'type': 'radio'},
            
            {'name': 'racs_low_galatic_region',
            'cols': ['RAJ2000', 'DEJ2000', 'maj', 'min', 'pa', 'total_flux', 'peak_flux', 'N/A', 'N/A', 's_code'],
            'type': 'radio'},
            
            {'name': 'racs_low_galatic_cut',
            'cols': ['RAJ2000', 'DEJ2000', 'maj', 'min', 'pa', 'total_flux', 'peak_flux', 'N/A', 'N/A', 's_code'],
            'type': 'radio'},
            
            {'name': 'spice-racs_catalog',
            'cols': ['RAJ2000', 'DEJ2000', 'maj_axis', 'min_axis', 'pa', 'total_I_flux', 'peak_I_flux', 'N/A', 'N/A', 's_code'],
            'drop_cols': ['source_id', 'spectral_index'],
            'type': 'radio'},
            
            {'name': 'TGSS',
            'cols': ['RAJ2000', 'DEJ2000', 'Maj', 'Min', 'PA', 'Total_flux', 'Peak_flux', 'N/A', 'N/A', 'Source_code'],
            'type': 'radio'},
            
            {'name': 'NVSS',
            'cols': ['RAJ2000', 'DEJ2000', 'MajAxis', 'MinAxis', 'PA', 'S1.4', 'N/A', 'N/A', 'N/A', 'N/A'],
            'type': 'radio',
            'flux_ratio': False},
            
            {'name': 'VLASS',
            'cols': ['RAJ2000', 'DEJ2000', 'Maj', 'Min', 'PA', 'Ftot', 'Fpeak', 'N/A', 'N/A', 'SCode'],
            'type': 'radio'},
                    
            {'name': 'SUMSS',
            'cols': ['RAJ2000', 'DEJ2000', 'Maj', 'Min', 'PA', 'St', 'Sp', 'N/A', 'N/A', 'Code'],
            'type': 'radio'},

            {'name': 'FIRST',
            'cols': ['RA', 'DEC', 'MAJOR', 'MINOR', 'POSANG', 'FINT', 'FPEAK',  'N/A', 'N/A', 'N/A'],
            'type': 'radio'}
    ]

    optiacal_inf_catalog = [
            {'name': 'WISE',
             'cols': ['ra', 'dec', 'WISE_maj','N/A', 'N/A', 'WISE_flux', 'N/A', 'N/A'],
             'type': 'infared',
            'flux_ratio': False}
    ]

    # bad code but we need the maj and flux empty column labels for it to pass as a source later on

    # catalog prep
    catalogs = load_and_prep_catalogs(radio_catalogs, base_dir=catalog_home_dir)
    regular_mals_catalog = catalogs['mals_all_bands']
    racs_galatic = catalogs['racs_low_galatic_region']
    racs_galatic_cut = catalogs['racs_low_galatic_cut']
    spice_racs_reducer= catalogs['spice-racs_catalog']
    tgss_reducer = catalogs['TGSS']
    nvss_reducer = catalogs['NVSS']
    vlass_reducer = catalogs['VLASS']
    sumss_reducer = catalogs['SUMSS']
    first_reducer = catalogs['FIRST']

    # print(nvss_reducer.catalog)
    # print(wise_reducer.catalog)

    
    for cat_year_idx, survey in enumerate(fermi_lat_surveys):
        '''TEMP'''
        # should of emphasized that the +1 to the cat idx before was temporary
        catalogs2 = load_and_prep_catalogs(optiacal_inf_catalog, base_dir=catalog_home_dir, cat_idx =cat_year_idx+3)
        # optical/inf call:
        wise_reducer = catalogs2['WISE']
        wise_reducer.catalog.remove_column('Source_Name')

        # fermi call
        survey_hud = fits.open(survey)
        Fermi_catalog = QTable.read(survey_hud, format='fits')
        print(f'====Analysis on {catalog_years[cat_year_idx]}-Yr-Fermi Catalog====')
        # make sure fermi is only calling unassociated sources
        # two step unknown sources only!
        class1_mask = np.isin(Fermi_catalog['CLASS1'], ['unk',''])
        if associated == False:
            Fermi_catalog = Fermi_catalog[class1_mask]
        else :
            Fermi_catalog = Fermi_catalog[~class1_mask]

        print(f'Fermi Unassociated Sources: {len(Fermi_catalog)}')
        fermi_name = 'FERMI'
        fermi_cataloging = Catalog(Fermi_catalog,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'],catalog_name=fermi_name, catalog_type='fermi' )
        '''FERMI OVERLAY- MALS, RACS, TGSS'''
        # since this is forced to be in a [] maybe fix it for single iteration if the user decides not to put it into a ls format?
        fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_cataloging, comparision_catalogs=[regular_mals_catalog, spice_racs_reducer, tgss_reducer,racs_galatic, racs_galatic_cut,
                                                                    nvss_reducer,vlass_reducer, sumss_reducer, first_reducer, wise_reducer])
        # fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_cataloging, comparision_catalogs=[regular_mals_catalog,])

        # fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_cataloging, comparision_catalogs=[regular_mals_catalog, uss_mals_catalog, css_mals_catalog])
        fermi_mals_overlay.is_in_ellipse_V2()
        print('fermi overlay done')
        fermi_ellipses_final, matched_comparision_catalogs, dict_comparisions_fermi, fermi_catalog, full_candidate_catalogs  = fermi_mals_overlay.matched_catalogs_V2(return_parent_catalogs=True)
        for index, name in enumerate(catalog_names):
            print(f'size of {name} matches:{len(matched_comparision_catalogs[index])}')


        '''SPW CROSSMATCHING MALS'''
        # changed fermi data call - 6/12
        #  changed again 6/26
        mal_cat = matched_comparision_catalogs[0]
        #    copy the source name  column for mals 
        '''REINSTATE AFTER'''
        # go through every single band for each unlinked object until all possible matches are created
        all_spw_labels = ['LSPW_0', 'LSPW_1', 'LSPW_2', 'LSPW_3',
                        'LSPW_4', 'LSPW_5', 'LSPW_6', 'LSPW_7',
                        'LSPW_8', 'LSPW_9', 'LSPW_10', 'LSPW_11',
                        'LSPW_12', 'LSPW_13', 'LSPW_14']
        all_spw_labels_check = len(all_spw_labels)
        all_checks = False
        matched_spws = []
        iteration = 0
        if mals_merged == True:
            # loop each spectral window starting from the lowest for source matches
            while all_checks == False:
                # applying spectral window matching 
                # while all_checks == False:
                # first iteration
                try:
                    spw_catalogs = {}
                    spw_catalogs[0] = mal_cat[mal_cat['spw_id'] ==all_spw_labels[iteration]]
                    mask_base_sources = np.isin(mal_cat['Indexes'], spw_catalogs[0]['Indexes'])
                    spw_catalogs[1] = mal_cat[~mask_base_sources]
                    spw_catalog_class_instances = {}
                except IndexError:
                    all_checks=True
                    continue
                
                for i in range(len(spw_catalogs.keys())):
                    spw_catalog_class_instances[i] = Catalog(spw_catalogs[i], ['RAJ2000', 'DEJ2000', 'maj_restoring_beam', 'min_restoring_beam', 'pa_restoring_beam', 'total_flux', 'peak_flux', 'spectral_index_spwfit', 'spectral_index_spwfit_e',  's_code', 'spw_id/WB'], catalog_type='radio')
                #TODO there are two empty fails here but I dont know why one still gets past this exception
                spw__all = CatalogOverlayer(base_catalog=spw_catalog_class_instances[0], comparision_catalogs=[spw_catalog_class_instances[1]])
                try:
                    spw__all.is_in_ellipse_V2()
                    spw_base_matched_catalog, spw_comparision_matched_catalog, dict_spws, spw_base_full_cat, spw_comparision_full_cats = spw__all.matched_catalogs_V2(return_parent_catalogs=True)
                except Exception as e:
                    print(e)
                    iteration +=1
                    all_spw_labels_check = len(all_spw_labels[iteration:])
                    continue
                    
                # spw__all.is_in_ellipse_V2()
                
                # spw_base_matched_catalog, spw_comparision_matched_catalog, dict_spws, spw_base_full_cat, spw_comparision_full_cats = spw__all.matched_catalogs_V2(return_parent_catalogs=True)
            
                mals_spectral_combined, no_base_match_sources = spectral_window_merging_V2(spw_base_full_cat, spw_comparision_full_cats[0], dict_spws[0])
                print('Spectral Indexing Complete')
                # data of current loop
                print(f'All matches : {all_spw_labels[iteration], len(mals_spectral_combined)}')
                print('=====')
                print(f'unassociated sources: {len(no_base_match_sources)}')
                print(f'Cross Spectral Window Matches {len(dict_spws[0].keys())}')
                # print('Spectral windows merge 0 and all complete')    
                
                # prepare for next loop by iterating through all 
                matched_spws.append(mals_spectral_combined)
                mal_cat = no_base_match_sources
                # all_spw_labels = all_spw_labels[1:]
                
                if all_spw_labels_check < 2:
                    all_checks=True
                    matched_spws.append(mal_cat)
                else:
                    # add the unassociated sources as well!
                    iteration +=1
                    all_spw_labels_check = len(all_spw_labels[iteration:]) 

                    continue
            mals_all_spws_merged = vstack(matched_spws)
            print(f'all matches {len(mals_all_spws_merged)}')
            
            '''I WOULD HAVE TO CALCULATE THE MALS SOURCES ALPHA VALUES HERE?!? Or do it after! the full generation of the catalogs'''
            # astropy fitting;chi-least-squared-analysis
            # append the slope as the spectral index and the chi squared as the error? per mals point 
            # option plot the sources in this case only!
            model = models.Polynomial1D(degree=1)
            fitter=fitting.LinearLSQFitter()
            freq_pattern     = r'ref_freq_LSPW'
            flux_pattern     = r'total_flux_LSPW'
            flux_err_pattern = r'total_flux_e_LSPW'

            mals_all_spws_merged = mals_all_spws_merged.filled(0)
            best_fit_slopes = []
            best_fit_intercepts = []
            chi_values = [] 
            
            # bayesian monte-carlo fitting of spectral indexes
            # spec_fitter_mc(mals_all_spws_merged)
            
            for mals_source in mals_all_spws_merged:
                
                # make all the (x) freqs in order and (y) fluxes in order
                freq_names = [string for string in mals_source.colnames if  freq_pattern in string]
                flux_names = [string for string in mals_source.colnames if flux_pattern in string]
                flux_err_names = [string for string in mals_source.colnames if flux_err_pattern in string]

                
        
                # regex call the flux names and make dimensionless
                flux_values = ([(src/u.mJy) for src in mals_source[flux_names] if src>0])
                flux_err_values = ([(src/u.mJy) for src in mals_source[flux_err_names] if src>0])
                freq_values = ([(src/u.MHz) for src in mals_source[freq_names] if src>0])
                
                try:
                    best_fit = fitter(model, freq_values, flux_values)
                    chi_value = calc_reduced_chi_square(best_fit(freq_values), freq_values, flux_values, flux_err_values, len(freq_values), 1 )
                    best_fit_slopes.append(best_fit.parameters[1] *u.dimensionless_unscaled)
                    best_fit_intercepts.append(best_fit.parameters[0] *u.dimensionless_unscaled)
                    chi_values.append(chi_value *u.dimensionless_unscaled)
                    # print(best_fit)
                except:
                    # for sources with only a single freq capture
                    best_fit_slopes.append(0.0 *u.dimensionless_unscaled)
                    best_fit_intercepts.append(0.0*u.dimensionless_unscaled)
                    chi_values.append(0.0*u.dimensionless_unscaled)
                    # come back to chi values they feel off!
                    
            mals_all_spws_merged.add_columns([best_fit_slopes,best_fit_intercepts,chi_values], names=('fit_alpha_slope', 'fit_start', 'chi_value'))
        
        else:
            # create a big dictionary to add all values to all the sources 
            while all_checks == False:
                # applying spectral window matching 
                # first iteration
                spw_catalogs = {}
                spw_catalogs[0] = mal_cat[mal_cat['spw_id'] ==all_spw_labels[iteration]]
                mask_base_sources = np.isin(mal_cat['Indexes'], spw_catalogs[0]['Indexes'])
                # repeats the sources here!
                spw_catalogs[1] = mal_cat[~mask_base_sources]
                spw_catalog_class_instances = {}
                
                for i in range(len(spw_catalogs.keys())):
                    spw_catalog_class_instances[i] = Catalog(spw_catalogs[i], ['RAJ2000', 'DEJ2000', 'maj_restoring_beam', 'min_restoring_beam', 'pa_restoring_beam', 'total_flux', 'peak_flux', 'spectral_index_spwfit', 'spectral_index_spwfit_e',  's_code', 'spw_id/WB'],
                                                             catalog_type='radio')
                    
                spw__all = CatalogOverlayer(base_catalog=spw_catalog_class_instances[0], comparision_catalogs=[spw_catalog_class_instances[1]])
                comparisions = spw__all.is_in_ellipse_V2()
                mals_dict = comparisions[0].ellipse_point_index                
                            
                mal_cat['spw_matches'] = mal_cat['source_name']
                for key in mals_dict.keys():
                    mal_cat['spw_matches'][mals_dict[key]] = mal_cat['source_name'][key]                
                
                # preps data for next iteration so there is no overlap/repitition
                mask_spw_matches = np.isin(mal_cat['spw_matches'], mal_cat['spw_matches'][list(mals_dict.keys())])
                no_base_match_sources = mal_cat[~mask_spw_matches]
                match_sources = mal_cat[mask_spw_matches]
                mal_cat = no_base_match_sources
                # splits mal into the 'matched' and 'unqiue sources sources again 
                if all_spw_labels_check < 2:
                    all_checks=True
                    matched_spws.append(match_sources)
                    matched_spws.append(no_base_match_sources)

                else:
                    matched_spws.append(match_sources)
                    iteration +=1
                    all_spw_labels_check = len(all_spw_labels[iteration:])
                    
                    # add the unassociated sources as well!
                    continue
               
            mals_all_spws_merged = vstack(matched_spws)
            print(f'all matches {len(mals_all_spws_merged)}')
            print(mals_all_spws_merged)
        # for the single loop of all spw windows with just the matching names
        #    get the global indexes for the dict and the key 
        #  matched_comparision_catalogs[matching_source][global_value_indexes] = matched_comparision_catalogs['Source_Name'][global_key_indexes]
        
        # plotting the chi values 
            # if i want to plot it add this 
            # print(freq_values)
            # print(flux_values)
            # print(chi_value)
            # plt.scatter(freq_values, flux_values, color='black')
            # plt.plot(freq_values, best_fit(freq_values), color='r')
            # plt.show()
            # add the best fit parameters 
            # add the spectral index 
            # if the source is unassociated add the spectral index of the source 
        # '''todo: change to be iterable!'''
        # done 6/30
        if len(matched_comparision_catalogs)==1:
            print('single comparision catalog')
            updated_comparision_catalogs = [mals_all_spws_merged]
            
        else:
            print('multiple comparisions catalogs detected')
            updated_comparision_catalogs = [mals_all_spws_merged, *matched_comparision_catalogs[1:]]

        for i in range(len(updated_comparision_catalogs)):
            updated_comparision_catalogs[i]['catalog_names'] = len(updated_comparision_catalogs[i])* [catalog_names[i]]
        
        # get global values,
        # make that the final dict to create the catalogs!
        # keeps only the values that are in the final matched global indexes
        global_dict_indexes = []
        for iteration in range(len(dict_comparisions_fermi)):
            dict_mal_femi_global_index = global_index(dict_comparisions_fermi[iteration], full_candidate_catalogs[iteration])
            dict_mal_femi_updated = {key: [value for value in value_list if value in (updated_comparision_catalogs[iteration]['Indexes'])] 
                                     for key, value_list in dict_mal_femi_global_index.items()
                                    }
            global_dict_indexes.append(dict_mal_femi_updated)

    
        # Configuration tags mapping strings
        search_term_z = ['Maj', 'MajAxis',  'MAJOR',  'maj_axis', 'maj', 'maj_restoring_beam', 'WISE_maj']
        search_term_flux = ['S1.4', 'FINT', 'Ftot', 'Stotal', 'total_flux', 'Total_flux', 'Flux', 'WISE_flux', 'total_I_flux', 'St']
        
        clean_comparison_catalogs = []
        for catalog in updated_comparision_catalogs:
            cat_dict = {}
            colnames = catalog.colnames
            # Look for the major axis column name in this catalog
            maj_col = None
            for c in colnames:
                # 1. First check if any of your search terms match the column name
                maj_col = next((c for c in colnames if c in search_term_z), None)
    
            
            for col in catalog.colnames:
                col_data = catalog[col]
                
                # If this is the major axis column, force it to degrees right now
                if col == maj_col and hasattr(col_data, 'to'):
                    try:
                        # Convert to degrees and extract the raw numeric array value
                        col_data = col_data.to(u.deg).value
                    except Exception:
                        # If it's already a raw number representing degrees, keep it as is
                        pass
                        
                # Safe extraction of arrays to strip Astropy structures/masks safely
                if hasattr(col_data, 'filled'):
                    if col_data.dtype.kind in {'i', 'u'}:
                        cat_dict[col] = col_data.filled(-999)
                    elif col_data.dtype.kind == 'f':
                        # FIX: Force float32 masked arrays to float64 to prevent mixed-type FITS errors
                        cat_dict[col] = col_data.filled(np.nan).astype(np.float64)
                    else:
                        cat_dict[col] = col_data.filled(np.nan)
                else:
                    np_arr = np.array(col_data)
                    if np_arr.dtype.kind == 'f':
                        # FIX: Force float32 standard arrays to float64 to prevent mixed-type FITS errors
                        cat_dict[col] = np_arr.astype(np.float64)
                    else:
                        cat_dict[col] = np_arr
                        
            clean_comparison_catalogs.append(cat_dict)
    
        total_sources = len(fermi_catalog)
        print(f"Starting cross-matching engine for {total_sources} sources...")
        '''Output the matched RACS, MALS tables, full grouped table, and hdu lists of individual fermi ellipses'''
        '''ADDING WISE AND GAIA'''
     
        # Per-source work is now cheap (no per-source Table construction, no
        # per-pair SkyCoord objects), so a plain sequential loop avoids the
        # process-pool dispatch/pickling overhead of one task per Fermi source.
        # processed_dicts = []
        if grouped_table_value==False:
            print('making closest source matches')
        elif grouped_table_value == True:
            print('making grouped table variant')
        hdu_list_holder = [fits.PrimaryHDU()]
        start_time = time.monotonic()
        last_report = start_time
    
        for ellipse_index in range(len(fermi_catalog)):
                # print(f'{i}/{len(fermi_catalog)}')
            result_dict = process_single_fermi(
                ellipse_index,
                fermi_catalog_sliced=fermi_catalog,
                global_dict_indexes=global_dict_indexes,
                updated_comparision_catalogs=updated_comparision_catalogs,
                search_term_z=search_term_z,
                search_term_flux=search_term_flux,
                grouped_table_value=grouped_table_value
            )
            # result dict MUST RETURN EITHER A EMPTY JUST FERMI OR ALL OF THE SOURCE WITH SPECTRAL INDEX IF VIABLE; cats are all dicts in a list so must make into a table then vstack
            if isinstance(result_dict, dict):
                result_dict = [result_dict]
                
            tables = []
            fermi_table = None
            
            # 1. First pass: Build the tables and identify the Fermi catalog template
            for cat in result_dict:
                if isinstance(cat, dict):
                    table = QTable(rows=[cat])
                    fermi_table = table  # Keep track of the table that holds the units
            
                else:
                    table = QTable(cat)
                tables.append(table)
    
            
            # 2. Second pass: Align units across all tables to match the Fermi units
            if fermi_table is not None:
                for colname in fermi_table.colnames:
                    fermi_unit = fermi_table[colname].unit
                    
                    # If the Fermi column has a unit (like deg, MeV, etc.)
                    if fermi_unit is not None:
                        for t in tables:
                            # If another catalog table has the same column name but lacks a unit
                            if colname in t.colnames and t[colname].unit is None:
                                # Upgrade the plain column to a Quantity with the matching unit
                                t[colname] = t[colname] * fermi_unit
    
            # it threw an error 
            for t in tables:
                if 'source_id' in t.colnames:
                    col = t['source_id']
                    
                    # If it's bytes or strings, clean it to standard string
                    if col.dtype.kind in ['S', 'U', 'O']:
                        # Decode if bytes, otherwise keep as string
                        t['source_id'] = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in col]
                    
                    # If it's integer or float, cast it to standard string
                    elif col.dtype.kind in ['i', 'u', 'f']:
                        t['source_id'] = col.astype(str)
            # 3. Stack safely without losing your units
            if len(tables) == 1:
                ellipse_grouped_table = tables[0]
            else:
                ellipse_grouped_table = vstack(tables, join_type='outer', metadata_conflicts='silent')
            name = ellipse_grouped_table[0]['Source_Name']
            print(f'Ellipse Matches: {len(ellipse_grouped_table)}')

            # DEBUG
    #                 # Loop through columns and force float columns to float64 to resolve mixed-type errors
    #             for col_name in ellipse_grouped_table.colnames:
    #                 col = ellipse_grouped_table[col_name]
    #                 print(f'col:{col_name}, type:{col.dtype.kind}')
    #                 # Check if the column contains floating-point numbers
    #                 if col.dtype.kind == 'f': 
    #                     ellipse_grouped_table[col_name] = col.astype('float64')
    #                 # If the column has a generic Object dtype ('O') or is mixed
    #                 if col.dtype.kind == 'O':
    #                     print(col)
    #                     # Replace masked/None values with empty strings safely
    #                     # cleaned_data = [str(x) if (x is not None and x is not masked) else '' for x in col]
    #                     # ellipse_grouped_table[colname] = cleaned_data
    # # 
                        
            
    
            # write to header 
            hdu = fits.BinTableHDU(ellipse_grouped_table, name=name)
            hdu_list_holder.append(hdu)
    
    
            # Time-based progress reporting instead of every-100 index checkpoints
            now = time.monotonic()
            if now - last_report >= 5:
                elapsed = now - start_time
                rate = (ellipse_index + 1) / elapsed
                print(f"Cross-Matched: {ellipse_index + 1}/{total_sources} sources "
                          f"({rate:.1f} sources/s, {elapsed:.0f}s elapsed)...")
                last_report = now
    
            
        print(f"\nCross-matching complete in {time.monotonic() - start_time:.0f}s. "
          "Instantiating masked master catalog matrix...")
        file_output_path = catalog_reduced_dir + folder_surveys[cat_year_idx]+file_marker+'_spws_merged_radio_fermi_tables.fits'
        hdu_master = fits.HDUList(hdu_list_holder)    
        hdu_master.writeto(file_output_path, overwrite=True)
        print(f'{folder_surveys[cat_year_idx]}yr-fermi catalogs complete')
    
        print(f"File written to: {file_output_path}")
            
    # # Export the final fully finished masked matrix to your FITS path
    # if pulsar_cat==True:
    #     final_output_path = catalog_reduced_dir + catalog_years[cat_year_idx] + 'yr_associated_pulsars.fits'
    #     ellipse_grouped_table.write(final_output_path, format='fits', overwrite=True)
    # else:
    #     final_output_path = catalog_reduced_dir + catalog_years[cat_year_idx] + 'yr_associated_ellipses_all_srcs.fits'
    #     ellipse_grouped_table.write(final_output_path, format='fits', overwrite=True)

if __name__ == '__main__':
    # print(timeit.timeit('main()', number=1))
    main(mals_merged=True, associated=False, grouped_table_value=True)
