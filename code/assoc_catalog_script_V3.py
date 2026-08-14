# creates the associated catalog (one big table where you collate all the sources per fermi ellipse!)
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
# from astropy.coordinates import SkyCoord
# import astropy.coordinates as coord
from astropy.io import fits
from astropy.table import vstack, hstack
# too hard to try and find all the quantities right now but need to fix later!
from astropy.table import QTable, Table, Column, MaskedColumn
from astropy.modeling import models, fitting
from astropy.coordinates import SkyCoord


import warnings
from astropy.utils.exceptions import AstropyWarning
from astropy.units import UnitsWarning
# Access astronomical databases
# from astroquery.vizier import Vizier
# For plots
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as ellipse
# Data handling
import numpy as np
import time

# frequencies should already be prepped
import glob
from astropy.table import QTable
import dask.dataframe as dd


'todo/issue; i could replace insertin every source by changing the multiwavecatalogs overlay to accept dicts with a container of catlaog info '
'otherwise i am doing an uncessary catalog name creation twic in this code!'
# calls all catalogs I specfiy with the parameters I set in the main file

from astropy.utils.exceptions import AstropyWarning

# 1. HIDE ALL WARNINGS GLOBALLY (Add these lines at the very top of your script)
warnings.filterwarnings('ignore', category=AstropyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message=".*merge.*")

# def get_clean_coords(table, ra_col, dec_col):
#     """Safely parses table coordinate strings or numbers into a SkyCoord object."""
#     if table[ra_col].dtype.kind in {'U', 'S', 'O'}:
#         try:
#             return SkyCoord(ra=table[ra_col], dec=table[dec_col], unit=(u.hourangle, u.deg))
#         except Exception:
#             return SkyCoord(ra=table[ra_col], dec=table[dec_col], unit=u.deg)
#     return SkyCoord(ra=table[ra_col], dec=table[dec_col], unit=u.deg)

def calculate_spectral_index(f1, f2, nu1, nu2):
    """Calculates alpha where Flux proportional to freq^alpha."""
    if f1 <= 0 or f2 <= 0 or nu1 == nu2 or nu1 <= 0 or nu2 <= 0:
        return np.nan
    return np.log(f1 / f2) / np.log(nu1 / nu2)


def load_and_prep_catalogs(catalog_configs, base_dir, run_flux_ratio=True):
    catalog_objects = {}
    for config in catalog_configs:
        name = config['name']
        cols = config['cols']
        type = config['type']
        drop_cols = config.get('drop_cols', [])  
        run_flux_ratio = config.get('flux_ratio', True)
        
        # Perform the glob search using the base file name
        search_pattern = f'{base_dir}{name}*.fits'
        file_matches = glob.glob(search_pattern)
        print(file_matches)

        if name =='WISE':
            file_matches.sort(reverse=False)
            file_matches = [file_matches[cat_idx]]
        
        if not file_matches:
            print(f'No files found matching pattern: {search_pattern}')
            continue
            
        # Catalog Class loop
        for file_path in file_matches:
            try:
                table = QTable.read(file_path, format='fits')
                if name == 'WISE':
                    for col in table.colnames:
                        if col not in ['wise_ang_sep', 'catalog_names']:
                            original_name = col.split('_', 1)[1]
                            table.rename_column(col, original_name)

                    table['ra'] = table['ra'] *u.deg
                    table['dec'] = table['dec'] *u.deg
                    table = table[~table['designation'].mask]
                                        
                # print(table.colnames)
                for col in drop_cols:
                    if col in table.colnames:
                        table.remove_column(col)
                catalog_instance = Catalog(table, cols, catalog_type=type)
                if run_flux_ratio:
                    catalog_instance.flux_ratio()
                
                catalog_objects[name] = catalog_instance
                print(f'Successfully processed catalog wrapper for: {name}')
                
            except Exception as e:
                print(f'Error executing catalog preparation on {file_path}: {e}')
                
    return catalog_objects

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


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
            all_candidate_rows.append(row)

    if not all_candidate_rows:
        return fallback_dict
    
    # print('check Step B')
    # Step B: check for nearby sources across catalogs and propagate/derive spectral index.
    # One vectorized SkyCoord + broadcasted separation matrix replaces building a
    # scalar SkyCoord per candidate and calling .separation() inside the O(k^2) loop.
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

                    # if calculated_alpha <= -10:
                    #     continue
                    # else:
                    row_i['spectral_index'] = calculated_alpha
                    row_j['spectral_index'] = calculated_alpha
                        # print('check')



    if grouped_table_value == True:
        catalog_groups = {}
        for row in all_candidate_rows:
            # to make them grouped
            row['Source_Name'] = fermi_source_dict['Source_Name']
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
        fermi_table = Table(rows=[dict(fermi_source_dict)])
        grouped_dict = []
        grouped_dict.append(fermi_table)
        # append the fermi source first!
        # then append the rest of the valid sources in it 
        for cat_base, rows in catalog_groups.items():
            # if cat_base == 'WISE':
            #     # print(rows)
            #     grouped_dict.append(rows)

            valid_spx_rows = [r for r in rows if not np.isnan(r['spectral_index'])]
            if not valid_spx_rows:
                continue
            else:
                valid_spx_table = Table(valid_spx_rows)

            grouped_dict.append(valid_spx_table)

        # # 2. Second pass: Align units AND normalize column classes safely
        # if fermi_table is not None:
        #     # Build a unified list of ALL column names present in ANY table in the group
        #     all_colnames = set()
        #     for t in grouped_dict:
        #         all_colnames.update(t.colnames)
        
        #     for colname in all_colnames:
        #         # Check if the Fermi table has a master unit for this column
        #         fermi_unit = fermi_table[colname].unit if colname in fermi_table.colnames else None
                
        #         for t in grouped_dict:
        #             if colname in t.colnames:
        #                 # --- STRINGS & TEXT HANDLING ---
        #                 # If the column holds text, bytes, or objects, it CANNOT be a Quantity.
        #                 # Force it to be a plain MaskedColumn to keep vstack happy.
        #                 if t[colname].dtype.kind in ['S', 'U', 'O']:
        #                     from astropy.table import MaskedColumn
        #                     if not isinstance(t[colname], MaskedColumn):
        #                         # Safely clean strings during conversion to avoid double-processing later
        #                         cleaned_data = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in t[colname]]
        #                         t[colname] = MaskedColumn(cleaned_data, name=colname)
                                
        #                 # --- NUMERIC HANDLING ---
        #                 else:
        #                     # Case A: Fermi dictates a specific physical unit (deg, MeV, etc.)
        #                     if fermi_unit is not None:
        #                         if t[colname].unit is None:
        #                             t[colname] = t[colname] * fermi_unit
                            
        #                     # Case B: No specific unit dictates this numeric column (e.g. flag counters, integers)
        #                     # Convert to dimensionless MaskedQuantity
        #                     else:
        #                         if not hasattr(t[colname], 'unit') or t[colname].unit is None:
        #                             t[colname] = t[colname] * u.dimensionless_unscaled
        
        # 3. Clean remaining numeric-to-string IDs safely
        for t in grouped_dict:
            if 'source_id' in t.colnames:
                col = t['source_id']
                # If it was a numeric column converted to string, make sure it is a MaskedColumn now
                if col.dtype.kind in ['i', 'u', 'f']:
                    from astropy.table import MaskedColumn
                    t['source_id'] = MaskedColumn(col.astype(str), name='source_id')
        
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

    # Step E: Adding background sources

def main(mals_merged = True, associated=True, file_marker='associated',pulsar_cat=False, grouped_table=False):
    if grouped_table == False:
        group_table_name = 'single'
    else:
        group_table_name = 'grouped'
    warnings.filterwarnings('ignore', category=AstropyWarning)
    warnings.simplefilter('ignore', category=UnitsWarning)
    # catalog call only 14 year associatd!
    # define home dir
    '''CATALOG CALLS'''
    print(os.getcwd()) 
    current_working_dir = os.getcwd()
    catalog_home_dir = current_working_dir + '/catalogs/'
    catalog_reduced_dir = current_working_dir + '/reduced_catalogs/'
    catalog_mals_all = current_working_dir + '/catalogs/mals_all_bands.fits'
    fermi_lat_surveys = glob.glob(catalog_home_dir + 'gll**35.fit')

    #the unassociated to associated surveys
    #fermi_lat_surveys = glob.glob(catalog_reduced_dir + 'unass_to_assoc*_yr_matches.fits')
    #fermi_lat_surveys.sort(reverse=False)

    print(fermi_lat_surveys)

    catalog_names = ['MALS_L', 'SPICE-RACS', 'TGSS', 'RACS_low','RACS_low', 'NVSS', 'VLASS', 'SUMSS', 'FIRST']
    catalog_years = ['14']
    #catalog_years  = ['8', '10', '12', '14']
    
    '''FERMI CATALOG CREATION'''
    # group sources into one large catalog for the associated values
    #fermi_lat_surveys = glob.glob('/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/gll**35.fit')

    
    mals_all_sources= QTable.read(catalog_mals_all, format='fits')
    # group the associated fermi sources with radio cmponents that are a match(closest) &
    # not the matches but in the error ellispe range (background)
    

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
    
    # 2. Call the function to construct and process your entire catalog library in 1 line    
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
    
    # regular_mals_catalog = copy(mals_catalog)
    
    for cat_year_idx, survey in enumerate(fermi_lat_surveys):
        # fermi call
        if pulsar_cat ==True:
            print(f'====Analysis on 12-Yr-Fermi Pulsar Catalog====')
            #match name to get err ellipse semi_maj and minor
            Fermi_catalog = QTable.read(catalog_home_dir + "pulsar_fermi3rd_cat.fits", format='fits')
            Fermi_catalog_error_ellipses = QTable.read(catalog_home_dir+ 'gll_psc_v31.fit', format='fits')
            Fermi_catalog_14 = QTable.read(catalog_home_dir +'gll_psc_v35.fit', format='fits')
            # print(Fermi_catalog_error_ellipses.colnames)

            '==============================================================pulsar to 12yr overlay=========================================================='

            Fermi_catalog['RAJD'] = Fermi_catalog['RAJD']*u.deg
            Fermi_catalog['DECJD'] = Fermi_catalog['DECJD']*u.deg
            Fermi_catalog['e_RAJ_stat'] = Fermi_catalog['e_RAJ_stat']*u.deg
            Fermi_catalog['e_DECJ_stat'] = Fermi_catalog['e_DECJ_stat']*u.deg
            
            #get the pulsar data
            fermi_name = 'Fermi_pulsar'
            fermi_cataloging_pulsar = Catalog(Fermi_catalog,['RAJD', 'DECJD','e_RAJ_stat', 'e_DECJ_stat', 'N/A'],catalog_name=fermi_name, catalog_type='xray' )

            #overlay with the general catalog fermi ellipses

            fermi_name_2 = 'FERMI_error_12yr'
            fermi_error_ellipses_cataloging = Catalog(Fermi_catalog_error_ellipses,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'],catalog_name=fermi_name_2, catalog_type='xray' )

            fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_error_ellipses_cataloging,
                                              comparision_catalogs=[fermi_cataloging_pulsar], add_ang_sep=True)

            fermi_mals_overlay.is_in_ellipse_V2()
            print('fermi12-pulsar overlay done')
            fermi_ellipses_final, matched_comparision_catalogs, dict_comparisions_fermi, fermi_catalog, full_candidate_catalogs  = fermi_mals_overlay.matched_catalogs_V2(return_parent_catalogs=True)
            
            #for all the matches combine the fermi ellipse relvant data
            #add the error ellipse from the correlated 12 year catalog
            # Fermi_catalog['Conf_95_SemiMajor'] = [0*u.deg] *len(Fermi_catalog)
            # Fermi_catalog['Conf_95_SemiMinor'] = [0*u.deg] *len(Fermi_catalog)
            # Fermi_catalog['Conf_95_PosAng']    = [0*u.deg] *len(Fermi_catalog)

            cat_1 = []
            cat_2 = []

            for key, value in dict_comparisions_fermi[0].items():
                #if len(value) >1:
                 #   closer_pulsar argmin(matched_comparision_catalogs[0]['ang_sep'][value])
                # instead of limiting do a hstack with the osurces

                cat_1.append(Fermi_catalog[value[0]])
                cat_2.append(Fermi_catalog_error_ellipses[key])
                # old way is to just keep this!
                # Fermi_catalog['Conf_95_SemiMajor'][value[0]] = Fermi_catalog_error_ellipses['Conf_95_SemiMajor'][key]
                # Fermi_catalog['Conf_95_SemiMinor'][value[0]] = Fermi_catalog_error_ellipses['Conf_95_SemiMinor'][key]
                # Fermi_catalog['Conf_95_PosAng'][value[0]]   = Fermi_catalog_error_ellipses['Conf_95_PosAng'][key]
            cat_1 = QTable(cat_1)
            cat_2 = QTable(cat_2)
            pulsar12yr_table = hstack([cat_2, cat_1])
            print(f'final size pulsars: {len(pulsar12yr_table)}')

            '==============================================================12 to 14yr overlay=========================================================='

            fermi_name = 'FERMI_pulsar'
            pulsar12yr_cataloging =  Catalog(pulsar12yr_table,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'],catalog_name=fermi_name, catalog_type='xray' )

            #overlay with the general catalog fermi ellipses
            fermi_name_2 = 'FERMI'
            fermi_ellipses_cataloging = Catalog(Fermi_catalog_14,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'],catalog_name=fermi_name_2, catalog_type='xray' )

            fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_ellipses_cataloging,
                                              comparision_catalogs=[pulsar12yr_cataloging], add_ang_sep=True)

            fermi_mals_overlay.is_in_ellipse_V2()
            fermi_ellipses_final, matched_comparision_catalogs, dict_comparisions_fermi_2, fermi_catalog, full_candidate_catalogs  = fermi_mals_overlay.matched_catalogs_V2(return_parent_catalogs=True)
            pulsar_cat_matched = full_candidate_catalogs[0]
            keys = list(dict_comparisions_fermi_2[0].keys())
            Fermi_catalog_14.remove_rows(keys)
            # if match remove the source it matches and then do a vstack
            cat_3 = []
            for key, value in dict_comparisions_fermi_2[0].items():
                #if len(value) >1:
                 #   closer_pulsar argmin(matched_comparision_catalogs[0]['ang_sep'][value])
                # instead of limiting do a hstack with the osurces
                cat_3.append(pulsar_cat_matched[value[0]])
        
            cat_3 = QTable(cat_3)
            cat_3['Extended_Source_Name'] = cat_3['Extended_Source_Name'].astype(str)

            common_columns = set(Fermi_catalog_14.colnames) & set(cat_3.colnames)
            
            # 2. Check for shape conflicts in those common columns
            for col in common_columns:
                # Compare dimensions past the row count dimension
                if Fermi_catalog_14[col].shape[1:] != cat_3[col].shape[1:]:
                    print(f"Shape conflict found in '{col}'. Dropping from Fermi_catalog_14 to force an empty/masked column.")
                    cat_3.remove_column(col)
                    
            Fermi_catalog = vstack([Fermi_catalog_14, cat_3], join_type='outer')
            print(f'Inital size 14yr Fermi_catalog: {len(Fermi_catalog_14)}')
            print(f'final size Fermi_catalog: {len(Fermi_catalog)}')
            print('fermi14-pulsar overlay done')
            
            # final fermi table to run through everything 
            fermi_cataloging = Catalog(Fermi_catalog,['RAJD', 'DECJD','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'],catalog_name=fermi_name, catalog_type='xray')

        else:
            survey_hud = fits.open(survey)
            Fermi_catalog = QTable.read(survey_hud, format='fits')
            print(f'====Analysis on {catalog_years[cat_year_idx]}-Yr-Fermi Catalog====')
            # make sure fermi is only calling unassociated sources
            # two step unknown sources only!
            clean_class1 = np.array([val for val in Fermi_catalog['CLASS1']])
            invalid_tags = {'', 'unk', 'unknown', 'unknow'}
            class1_valid = np.array([val not in invalid_tags for val in clean_class1])
            if associated == True:
                Fermi_catalog = Fermi_catalog[class1_valid]
                print(f"Total valid associated sources: {len(Fermi_catalog)}")
            else:
                Fermi_catalog = Fermi_catalog[~class1_valid]
                print(f"Total valid unassociated sources: {len(Fermi_catalog)}")
        
                # Normalize all float columns to float64 up front. Mixed float32/float64
                # columns of the same name across per-source dicts (matched vs. fallback
                # rows) is what causes the FITS write to fail at the very end of main() -
                # fixing it here means every downstream copy inherits a consistent dtype.
            for col_name in Fermi_catalog.colnames:
                col = Fermi_catalog[col_name]
                if col.dtype.kind == 'f' and col.dtype != np.float64:
                    Fermi_catalog[col_name] = col.astype(np.float64)

                fermi_name = 'FERMI'
                fermi_cataloging = Catalog(Fermi_catalog,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'],catalog_name=fermi_name, catalog_type='fermi' )
        '''FERMI OVERLAY- MALS, RACS, TGSS, VLASS, NVSS, SUMSS, FIRST'''
        # since this is forced to be in a [] maybe fix it for single iteration if the user decides not to put it into a ls format?
        fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_cataloging, comparision_catalogs=[regular_mals_catalog, spice_racs_reducer, tgss_reducer,racs_galatic, racs_galatic_cut,
                                                                    nvss_reducer,vlass_reducer, sumss_reducer, first_reducer],add_ang_sep=True)
        # fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_cataloging, comparision_catalogs=[regular_mals_catalog,])
        fermi_mals_overlay.is_in_ellipse_V2()
        print('fermi overlay done')
        fermi_ellipses_final, matched_comparision_catalogs, dict_comparisions_fermi, fermi_catalog, full_candidate_catalogs  = fermi_mals_overlay.matched_catalogs_V2(return_parent_catalogs=True)
        
        print(matched_comparision_catalogs[1]['ang_sep'])
        
        # size check
        rac_check = 0
        for index, catalog in enumerate(matched_comparision_catalogs):
            if rac_check==1:
                continue
            elif catalog_names[index] == 'RACS_low':
                print(f'size of {catalog_names[index]}:{len(catalog)+len(matched_comparision_catalogs[index+1])}')
                rac_check+=1
            else:
                print(f'size of {catalog_names[index]}:{len(catalog)}')
            
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
                #try-except in the case of the max iterable being empty in cases like the pulsar matching (where the mals points are funky and could be none!)
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
                   
                spw__all = CatalogOverlayer(base_catalog=spw_catalog_class_instances[0], comparision_catalogs=[spw_catalog_class_instances[1]])
                try:
                    #TODO there are two empty fails here but I dont know why one still gets past this exception
                    spw__all.is_in_ellipse_V2()
                    spw_base_matched_catalog, spw_comparision_matched_catalog, dict_spws, spw_base_full_cat, spw_comparision_full_cats = spw__all.matched_catalogs_V2(return_parent_catalogs=True)
                except Exception as e:
                    print(e)
                    iteration +=1
                    all_spw_labels_check = len(all_spw_labels[iteration:])
                    continue
                

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
                    best_fit_slopes.append(np.nan *u.dimensionless_unscaled)
                    best_fit_intercepts.append(np.nan*u.dimensionless_unscaled)
                    chi_values.append(np.nan*u.dimensionless_unscaled)
                    # come back to chi values they feel off!
            try:
                mals_all_spws_merged.add_columns([best_fit_slopes,best_fit_intercepts,chi_values], names=('spectral_index', 'fit_start', 'chi_value'))
            except ValueError:
                mals_all_spws_merged.replace_column('spectral_index', best_fit_slopes)
                mals_all_spws_merged.replace_column('fit_start', best_fit_intercepts)
                mals_all_spws_merged.replace_column('chi_value', chi_values)
                

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
                    spw_catalog_class_instances[i] = Catalog(spw_catalogs[i], ['RAJ2000', 'DEJ2000', 'maj_restoring_beam', 'min_restoring_beam', 'pa_restoring_beam', 'total_flux', 'peak_flux', 'spectral_index_spwfit', 'spectral_index_spwfit_e',  's_code', 'spw_id/WB'], catalog_type='radio')
                    
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

        '''Other Spectral Matching'''
        # match the spectral indexes of unassociated mals and all the other sources 
        
        # the order is dictated by the order of the overlay
        if len(matched_comparision_catalogs)==1:
            print('single comparision catalog')
            updated_comparision_catalogs = [mals_all_spws_merged]
            
        else:
            print('multiple comparisions catalogs detected')
            updated_comparision_catalogs = [mals_all_spws_merged, *matched_comparision_catalogs[1:]]

        # update each source with attributes here 
        for i in range(len(updated_comparision_catalogs)):
            updated_comparision_catalogs[i]['catalog_names'] = len(updated_comparision_catalogs[i])* [catalog_names[i]]
            
        # get global values,
        # make that the final dict to create the catalogs!
        # keeps only the values that are in the final matched global indexes
        global_dict_indexes = []
        for iteration in range(len(dict_comparisions_fermi)):
            # called it dict_mal_fermi but its not just fermi that is compared its all catalogs
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

    # Per-source work is now cheap (no per-source Table construction, no
    # per-pair SkyCoord objects), so a plain sequential loop avoids the
    # process-pool dispatch/pickling overhead of one task per Fermi source.
    processed_tables = []
    start_time = time.monotonic()
    last_report = start_time
    if grouped_table==False:
        print('making closest source matches')
                
        for source_idx in range(total_sources):
            result_dict = process_single_fermi(
                source_idx,
                fermi_catalog_sliced=fermi_catalog,
                global_dict_indexes=global_dict_indexes,
                updated_comparision_catalogs=clean_comparison_catalogs,
                search_term_z=search_term_z,
                search_term_flux=search_term_flux,
                grouped_table_value=grouped_table
            )
            processed_tables.append(result_dict)
            
    elif grouped_table == True:
        print('making grouped table variant')
        
        for source_idx in range(total_sources):
            result_dict = process_single_fermi(
                source_idx,
                fermi_catalog_sliced=fermi_catalog,
                global_dict_indexes=global_dict_indexes,
                updated_comparision_catalogs=clean_comparison_catalogs,
                search_term_z=search_term_z,
                search_term_flux=search_term_flux,
                grouped_table_value=grouped_table
            )
            # result_table = vstack(result_dict)
            if len(result_dict) == 1:
                    result_table = result_dict[0]
            else:
                result_table = vstack(result_dict, join_type='outer', metadata_conflicts='silent')
            processed_tables.append(result_table)
    
            # Time-based progress reporting instead of every-100 index checkpoints
            now = time.monotonic()
            if now - last_report >= 5:
                elapsed = now - start_time
                rate = (source_idx + 1) / elapsed
                print(f"Cross-matching: {source_idx + 1}/{total_sources} sources "
                          f"({rate:.1f} sources/s, {elapsed:.0f}s elapsed)...")
                last_report = now
                    
    print(f"\nCross-matching complete in {time.monotonic() - start_time:.0f}s. "
              "Instantiating vstacking")
    # FIX 1: Enforce masked=True on initialization. 
    # This automatically turns inconsistent keys into standardized MaskedColumns.
    ellipse_grouped_table = vstack(processed_tables)
        # Loop through columns and force float columns to float64 to resolve mixed-type errors
    for col_name in ellipse_grouped_table.colnames:
        col = ellipse_grouped_table[col_name]
        print(f'col:{col_name}, type:{col.dtype.kind}')
        # Check if the column contains floating-point numbers
        if col.dtype.kind == 'f': 
            ellipse_grouped_table[col_name] = col.astype('float64')

    
    # Export the final fully finished masked matrix to your FITS path
    if pulsar_cat==True:
        final_output_path = catalog_reduced_dir + group_table_name + f'_{catalog_years[cat_year_idx]}' + 'yr_associated_pulsars.fits'
        ellipse_grouped_table.write(final_output_path, format='fits', overwrite=True)
    else:
        final_output_path = catalog_reduced_dir + group_table_name+ f'_{catalog_years[cat_year_idx]}' + 'yr_associated_ellipses_all_srcs.fits'
        ellipse_grouped_table.write(final_output_path, format='fits', overwrite=True)

    print(f"Successfully finalized catalog architecture! File written to: {final_output_path}")
    print(f"{catalog_years[cat_year_idx]}yr-fermi catalogs complete")

if __name__ == '__main__':
    # print(timeit.timeit('main()', number=1))
    main(mals_merged=True, associated=True, pulsar_cat=False, grouped_table=False)
