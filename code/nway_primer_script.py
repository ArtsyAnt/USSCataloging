# takes input ellipse(s) from hdulist
import glob
from astropy.io import fits
from astropy.table import vstack
from astropy.table import QTable, Column, MaskedColumn
from astropy.table import Table
import os
# goal of this script is just to find the associations between each DR4/5 and ALL of its associates
# in doing so it will run the fermi overlay script

from multiwavecataloging import Catalog
from multiwavecataloging import CatalogOverlayer
from multiwavecataloging import fermi_plot
from multiwavecataloging import spectral_window_merging_V2
from multiwavecataloging import global_index
from multiwavecataloging import calc_reduced_chi_square

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import hstack, vstack
# from astropy.table import vstack

# Access astronomical databases

from astroquery.vizier import Vizier
from astropy.io.votable import parse
# Data handling
import numpy as np
# import pymc as pm
from nwaylib import nway_match
from astropy.coordinates import ICRS, FK5
# warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
from astropy.units import UnitsWarning

# this is ref from nway-apitest.py

def table_from_catalog(catalog, name, poserr_value=None, area=None, magnitude_columns=[]):
    fits_table = catalog
    table_name = name
    ra = fits_table['RA']
    dec = fits_table['DEC']
    poserr = fits_table['pos_err']
    # magnitude columns
    mags = []
    maghists = []
    magnames = []
    #for mag in magnitude_columns:
    for col_name, magfile in magnitude_columns:
        assert col_name in fits_table.dtype.names
        mag_all = fits_table[col_name]
        # mark -99 as undefined
        mag_all[mag_all == -99] = np.nan
        mags.append(mag_all)
        magnames.append(col_name)
        if magfile == 'auto':
            maghists.append(None)
        # 1d prior!
        else:
            bins_lo, bins_hi, hist_sel, hist_all = np.loadtxt(magfile).transpose()
            maghists.append((bins_lo, bins_hi, hist_sel, hist_all))

    # need to return other columns!!!!! n
    
    # print(magnames)
    # print(maghists)
    return dict(name=table_name, ra=ra, dec=dec, error=poserr, area=area, mags=mags, maghists=maghists, magnames=magnames, original_table=catalog)

	# area in square degrees
	# error in arcsec
	# ra/dec in degrees
	# mag: column of something
	# maghists: either (bin, sel, all) tuple or None (for auto)
 
# creates an offset for the fermi ellipses
# creates an offset for the fermi ellipses
def create_fake_catalog(catalog):
    # make a random ra and dec offset that is X greater than the 95% positional uncertainities! 
    # call position parameters Safely extract scalars
    ra = catalog['RA'][0] if hasattr(catalog['RA'], '__len__') else catalog['RA']
    dec = catalog['DEC'][0] if hasattr(catalog['DEC'], '__len__') else catalog['DEC']
    
    ra_err = catalog['Conf_95_SemiMajor'][0] if hasattr(catalog['Conf_95_SemiMajor'], '__len__') else catalog['Conf_95_SemiMajor']
    dec_err = catalog['Conf_95_SemiMinor'][0] if hasattr(catalog['Conf_95_SemiMinor'], '__len__') else catalog['Conf_95_SemiMinor']
    print(ra)
    print(dec)
    
    # Handle NaN errors safely by substituting a tiny offset in the same direction
    if np.isnan(ra_err): ra_err = 0.001
    if np.isnan(dec_err): dec_err = 0.001

    # --- REGION-SAFE BOUNDS CHECK ---
    # Dynamically sort the bounds so 'low' is always mathematically smaller than 'high'
    ra_bound_1 = ra + ra_err * 0.5
    ra_bound_2 = ra + ra_err
    ra_low, ra_high = min(ra_bound_1, ra_bound_2), max(ra_bound_1, ra_bound_2)

    dec_bound_1 = dec + dec_err * 0.5
    dec_bound_2 = dec + dec_err
    dec_low, dec_high = min(dec_bound_1, dec_bound_2), max(dec_bound_1, dec_bound_2)
    # ---------------------------------

    # Generate random coordinates using the safe intervals
    random_ra  = np.random.uniform(ra_low, ra_high)
    random_dec = np.random.uniform(dec_low, dec_high)
    
    print(f"Generated Fake RA: {random_ra}")
    print(f"Generated Fake DEC: {random_dec}")

    # update coordinates
    fake_catalog = catalog.copy()
    fake_catalog['RA'] = random_ra
    fake_catalog['DEC'] = random_dec
    fake_catalog['catalog_names'] = 'FAKE_' + fake_catalog['catalog_names']
    return fake_catalog

    
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
        
        if not file_matches:
            print(f'No files found matching pattern: {search_pattern}')
            continue
            
        # Catalog Class loop
        for file_path in file_matches:
            try:
                table = QTable.read(file_path, format='fits')
                for col in drop_cols:
                    if col in table.colnames:
                        table.remove_column(col)
                catalog_instance = Catalog(table, cols, catalog_name = name, catalog_type=type)
                if run_flux_ratio:
                    catalog_instance.flux_ratio()
                
                catalog_objects[name] = catalog_instance
                print(f'Successfully processed catalog wrapper for: {name}')
                
            except Exception as e:
                print(f'Error executing catalog preparation on {file_path}: {e}')
    return catalog_objects  
# could be bad but just a basic uncertainity for the catalogs
# change to just ra uncertainity if necessary!
def pos_unc(catalog):
    match catalog['catalog_names'][0]:
        case 'FERMI':
            ra_err = (catalog['Conf_95_SemiMajor']*u.deg).to(u.arcsec)
            dec_err = (catalog['Conf_95_SemiMinor']*u.deg).to(u.arcsec)
            pa = (catalog['Conf_95_PosAng']*u.deg)

        case 'MALS_L'|'RACS_low':
            ra_err = catalog['ra_mean_e']
            dec_err = catalog['dec_mean_e']
            pa = catalog['pa_e']*u.deg

        case'FIRST':
            # unique have to derive it a differnet wya
            ra_err = 0
            dec_err= 0
            pa = 0
            
        case 'TGSS':
            ra_err = catalog['E_DEC']
            dec_err = catalog['E_RA']
            pa = catalog['E_PA']*u.deg
            
        case 'SUMSS'|'NVSS':
            ra_err = catalog['e_RAs']
            dec_err = catalog['e_DEs']
            pa = 0

        case 'VLASS':
            ra_err = catalog['e_RAdeg']
            dec_err = catalog['e_DEdeg']
            pa = catalog['e_PA']*u.deg
            

        case 'WISE':
            ra_err = catalog['sigra']
            dec_err = catalog['sigdec']
            pa = 0

        case 'PANSTARRS':
            ra_err = catalog['e_RAJ2000']
            dec_err = catalog['e_DEJ2000']
            pa = 0

        # case 'GAIA':
        #     ra_err = catalog['RA_e']
        #     dec_err = catalog['DEC_e']
        #     pa = 0
            
        # vizier catalog search
        case _:
            pass

    angle = np.radians(pa)
    # idk about the angle but the other asepcts are fine
    positional_uncertainty = np.sqrt((ra_err**2)*(np.cos(angle)**2) + dec_err**2)
    if np.isnan(positional_uncertainty).any():
        print(f'error: positionaly uncertainity nan for {catalog['catalog_names'][0]} catalog')
        max_uncer = np.nanmax(positional_uncertainty)
        positional_uncertainty = [max_uncer] *len(catalog)
        return positional_uncertainty
    else:
        return positional_uncertainty

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
        # coords = {"obs_dim": np.arange(len(log_freq)), "log_freq_coord": log_freq}
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
            # Jy mean prior
            # should this be the deterministic value?
            c = pm.Normal('c', c_log_value, sigma=2)
            
            # the expected mean value for the log space
            # since it is not constant it is only acceped under this changed module
            spec_flux_density = pm.Deterministic(r'$S_{\nu}$', c + alpha * log_freq, dims='obs_dim') 
            
            # liklihood function (for the bayesian analysis, normal models residulas)
            # check if this is actually accurate!
            obs_spec_flux_density = pm.Normal(r'Observed $S_{\nu}$', mu=spec_flux_density, sigma=log_flux_error, observed=log_flux, dim='obs_dim')
            # sampler
            print(spectral_model.debug())
            sample = pm.sample(draws=1000, tune=1000,  target_accept=0.9)
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
    
    # choose if you want to call this in a terminal or run it in the file (recommended call in terminal if doing single source!)
    # design to run either all sources in a given catalog year or (x amount)
    # can calibrate p_any w/offset
    # keeps the intermediary catalogs 
    # if no fermi_names are specificed (from the fermi source_name descriptor) it will do a nway match for all sources
    # to add vizier call
def main(catalog_year ='14',match_type='mag-prior', prior_input = None,
         terminal_call =True, offset_calibration=False, keep_catalogs=True,
         fermi_names=None, 
         ):
    
    warnings.filterwarnings('ignore', category=AstropyWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message=".*merge.*")

    current_working_dir = os.getcwd()
    catalog_home_dir = current_working_dir + '/catalogs/'
    catalog_reduced_dir = current_working_dir + '/reduced_catalogs/'
    catalog_mals_all = current_working_dir + '/catalogs/mals_all_bands.fits'
    if match_type =='mag-prior':
        match_type_path = 'mag_prior_all_nway_matches/'
    
    elif match_type == 'distance':
        match_type_path = 'distance_all_nway_matches/'

    fermi_list_combined = catalog_reduced_dir + catalog_year +'_yr/' + 'unassociated_spws_merged_radio_fermi_tables.fits'
    # maybe make a easy way to click and choose from a list!
    
    hdu_table = fits.open(fermi_list_combined,  memmap=False)
    # source_name = '4FGL J0041.3-0048'
    source_names = [hdu.name for hdu in hdu_table]
    source_names = source_names[1:]
    catalog_size = len(source_names)

    # only look for sources that I havent already dont 
    if keep_catalogs == True:
        path = catalog_reduced_dir+match_type_path+catalog_year+ '/4FGL*'
        files_to_skip =  glob.glob(path)
        source_names_to_skip = set()
        for filepath in files_to_skip:
            filename = os.path.basename(filepath)
            # Example:
            # 4FGL J0008.9+2509
            if '_table_' in filename:
                source_name = filename.split('_table_', 1)[0]
                source_names_to_skip.add(source_name)
    
        # Remove completed sources regardless of their position
        source_names = [
            name for name in source_names
            if name not in source_names_to_skip]
        skip_num = len(source_names_to_skip)
        
        print(path)
        print(f'{skip_num} sources skipped')

        # source_names = source_names[skip_num:]
        # print(files_to_skip)
        # removes x file never input lol
    else:
        files_to_skip = []
        skip_num = len(files_to_skip)

        
    for source_index, source_name in enumerate(source_names):
        print(f'======{source_name}======')
        ellipse = QTable(hdu_table[source_name].data)
        # TEMP
        ellipse['Source_Name'][1:] = ''
        # hardcoded skip so that we only focus on the smaller ellipse matches
        if len(ellipse) > 450:
            continue
        # print(ellipse)
        # ellipse.remove_column('DEC_Counterpart')
        # best to rename all files to work with the nway internal and extrenal call

        catalogs = []
        catalog_names = list(set(ellipse['catalog_names']))
        for name in catalog_names: 
            table_mask = np.isin(ellipse['catalog_names'], name)
            catalog = ellipse[table_mask]
            # if name =='WISE':
            #     if np.isnan(catalog['RA']):
            #         continue
                # catalog.rename_column('ra', 'RA')
                # catalog.rename_column('dec', 'DEC')
                
        
            # remove the empty columns created when merging the base radio catalogs! add more when 
    
            cols_to_remove = []
            for col_name in catalog.colnames:
                col = catalog[col_name]
                
                # 1. Check if the entire column consists of blank/empty strings or formatting artifacts
                is_empty_string = np.all(np.isin(col, ['--', '[]', '', ' ']))
                
                # 2. Check if the entire column consists of numerical NaNs
                # Wrapped in a try/except because np.isnan crashes on string data types
                try:
                    is_all_nan = np.all(np.isnan(col))
                except TypeError:
                    is_all_nan = False
                    
                # 3. Check if the entire column is masked out (empty fields from an Astropy table merge)
                is_all_masked = hasattr(col, 'mask') and np.all(col.mask)
                
                # If ANY of these conditions are completely true for the column, flag it for deletion
                if is_empty_string or is_all_nan or is_all_masked:
                    cols_to_remove.append(col_name)
            
            # Ensure essential identification columns are never dropped
            protected_cols = {'CLASS1', 's_code', 'ASSOC1', 'Source_Name',  'Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'} 
            cols_to_remove = [col for col in cols_to_remove if col not in protected_cols]
            
            # print('removing columns:')
            # print(cols_to_remove)
            catalog.remove_columns(cols_to_remove) 
            # since i didnt properly transform all the catalogs before properly need to make sure that is a core part of catalog prep!
            try:
                catalog.rename_column('RAJ2000', 'RA')
            except:
                pass
            try:
                catalog.rename_column('DEJ2000', 'DEC')
            except:
                pass
            try:
                catalog.rename_column('ra', 'RA')
            except:
                pass
            try:
                catalog.rename_column('dec', 'DEC')
            except:
                pass
            # get the positional uncertanity for each catalog 
            positional_uncer = pos_unc(catalog)
            catalog['pos_err'] = positional_uncer
            # print(catalog)
                
            # used to identify the fermi catalog I could probably combine both the for loops since they do the same process of iterating between catalogs 
            if catalog['Source_Name'][0] == source_name:

                # print(catalog)
                # create the catalog reducer file for the other overlay
                fermi_catalog_reducer = Catalog(catalog, ['RA', 'DEC','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'],catalog_name='FERMI', catalog_type='xray' )
                offset_catalog = create_fake_catalog(catalog)
                catalogs.append(offset_catalog)
                catalog['ID'] = 1
                ra = catalog['RA'][0]
                dec =  catalog['DEC'][0]
                semi_major = catalog['Conf_95_SemiMajor'][0]
                semi_minor = catalog['Conf_95_SemiMinor'][0]
            else:
                print('non-fermi catalog; removing additional columns')
                un_protected_cols = {'CLASS1',  'ASSOC1', 'Source_Name'}
                catalog.remove_columns(un_protected_cols)
            catalogs.append(catalog)

        #spec_match mals catalog here
        # all catalogs will match the same area!
        radius_search = (semi_major*u.deg).to(u.arcsec)
        mag_search_radius = (semi_major*u.deg).to(u.arcsec) * 0.5
        area = np.pi * semi_major * semi_minor
        print(f'search radius (arcsec): {radius_search}')
        print(f'recommended mag radius (arcsec): {mag_search_radius}')
        
        primed_catalogs =[]
        catalog_paths = []
        # if; the writing for the files should be here
        for catalog in catalogs:
            name = list(set(catalog['catalog_names']))
            print(name)
            catalog.meta['EXTNAME'] = name[0]
            catalog.meta['SKYAREA'] = area  
            catalog_path= catalog_reduced_dir + 'send_test/'+ f'{name[0]}_catalog.fits'  
            # catalogs.append(catalog)
            # catalog_paths.append(catalog_path)
            # catalog.write(catalog_path, format='fits', overwrite='True')
        
        # terminal prep
        if terminal_call == True:
            for index, catalog in enumerate(primed_catalogs):
                catalog.write(catalog_path[index], format='fits', overwrite='True')
                # print(f"format for nway terminal call:\n python ~nway.py {catalog.meta['EXTNAME']['FERMI']} :pos_err {catalog.meta['EXTNAME']['FERMI']} --out={catalog['FERMI']['source_name']}_bay_.fits --radius{radius_search} --mag CATNAME:COL prior_type --mag-radius input")
        
        # script call
        else:
            # create an array for catalogs, magnitude columns 
             # create an array for catalogs, magnitude columns
            primed_tables = []
            
            # Track names to prevent appending duplicate catalog runs (e.g., dual WISE tables)
            seen_catalog_names = set()
        
            for catalog in catalogs:
                catalog = catalog.filled(0)
                name = catalog.meta['EXTNAME']
                skyarea = catalog.meta['SKYAREA']
                catalog = Table(catalog.as_array())
                
                # Skip this catalog if we already processed a valid instance of it in this run
                if name in seen_catalog_names:
                    print(f"Skipping duplicate catalog entry for: {name}")
                    continue
        
                primed_table = None # Reset placeholder
        
                match name:
                    case 'FERMI'|'GAIA'|'PANSTARRS':
                        primed_table = table_from_catalog(catalog, name, area=skyarea)
        
                    case 'MALS_L'|'RACS_low'|'VLASS'|'SUMSS'|'FIRST'|'NVSS'|'TGSS':
                        if match_type == 'distance':
                            primed_table = table_from_catalog(catalog, name, area=skyarea)
                        elif match_type == 'mag-prior':
                            priors = glob.glob(current_working_dir + '/priors/' + name + '-*.txt')
                            # Correctly map flat tuples of (column_name, path)
                            
                            magnitude_columns_input = []
                            for prior_path in priors:
                                # print(prior_path)
                                # Extract clean column header name from the file name
                                # THIS CAN INCREASE DEPENDING ON YOUR FILE PATH BE WARY!
                                col_name = prior_path.split('-', 3)[2]
                                # print(col_name)
                                if col_name in catalog.colnames:
                                    magnitude_columns_input.append((col_name, prior_path))
                            # print(magnitude_columns_input)
                            
                            try:
                                primed_table = table_from_catalog(catalog, name, area=skyarea, magnitude_columns=magnitude_columns_input)
                            except Exception as e:
                                print(f'Error generating prior table for {name}, falling back to distance match: {e}')
                                primed_table = table_from_catalog(catalog, name, area=skyarea)

                    case 'WISE':
                        if match_type == 'distance':
                            primed_table = table_from_catalog(catalog, name, area=skyarea)
                        elif match_type == 'mag-prior':
                            primed_table = table_from_catalog(catalog, name, area=skyarea)
        
                    case 'FAKE_FERMI':
                        catalog_path = catalog_reduced_dir + f'{match_type_path}/' + f'{catalog_year}'+'/fakes/' + f'{source_name}_fermi_table.fits'
                        catalog.write(catalog_path, format='fits', overwrite=True)
                        # primed_table = table_from_catalog(catalog, name, area=skyarea)
        
                # Only append to tables if a valid wrapper structure was built, and track the name
                if primed_table is not None:
                    primed_tables.append(primed_table)
                    seen_catalog_names.add(name)

        
            print(f"Total Unique Catalogs Staged: {len(primed_tables)}")
        
            if len(primed_tables) <= 1:
                print('Insufficient matching catalogs found (1 or 0)')
                print(f'{source_index+1+skip_num}/{catalog_size}')
                continue
        
            for index, table in enumerate(primed_tables):
                if table['name'] == 'FERMI':
                    if index > 0:
                        fermi_table = primed_tables.pop(index)
                        primed_tables.insert(0, fermi_table)
                        break
                    else:
                        break

        
            # --- CLEAN CORRUPTED DATA BEFORE MATCHING ---
            for table_dict in primed_tables:
                orig_table = table_dict['original_table']
                
                # 1. Clean Position Errors: Replace NaN, 0, or negative errors with a safe floor (e.g., 0.1 arcseconds)
                if 'pos_err' in orig_table.colnames:
                    bad_errors = (np.isnan(orig_table['pos_err'])) | (orig_table['pos_err'] <= 0)
                    if np.any(bad_errors):
                        print(f"WARNING: Found {np.sum(bad_errors)} bad/zero positional errors in {table_dict['name']}. Setting to 0.5 arcsec floor.")
                        orig_table['pos_err'][bad_errors] = 0.5
                        
                # 2. Sync dictionary keys: Update the array nway actually reads
                table_dict['error'] = orig_table['pos_err']
        
                # 3. Clean Coordinates: Ensure no RA or DEC fields are missing/NaN
                if 'RA' in orig_table.colnames and 'DEC' in orig_table.colnames:
                    bad_coords = np.isnan(orig_table['RA']) | np.isnan(orig_table['DEC'])
                    if np.any(bad_coords):
                        print(f"CRITICAL: Found {np.sum(bad_coords)} NaN coordinates in {table_dict['name']}. Removing corrupted rows.")
                        # Filter rows completely out of the internal match arrays
                        valid_mask = ~bad_coords
                        table_dict['ra'] = table_dict['ra'][valid_mask]
                        table_dict['dec'] = table_dict['dec'][valid_mask]
                        table_dict['error'] = table_dict['error'][valid_mask]
                        table_dict['original_table'] = orig_table[valid_mask]

            # Execute matching with protective error management
            try:
                # need to sort primed table first 
                df_result  = nway_match(
                    primed_tables,
                    prior_completeness=1,
                    match_radius = radius_search.value, 
                    store_mag_hists = False,
                    mag_include_radius = mag_search_radius.value, 
                )
                result = Table.from_pandas(df_result)

            except Exception as e:
                print(f"CRITICAL: nway_match failed execution on source {source_name}: {e}")
                print(f'{source_index+1+skip_num}/{catalog_size}')
                continue # <--- THIS STOPS UNBOUNDLOCALERROR FROM REACHING THE NEXT LINE

            # We will build a list of tables to stitch horizontally alongside the nway results
            final_tables_to_combine = [result]

            fermi_match = primed_tables[0]['original_table']
            results_size = len(result)
            large_table = vstack([fermi_match] * results_size)
            for col in large_table.colnames:
                    large_table.rename_column(col, f"FERMI_{col}")
                    
            
            final_tables_to_combine.append(large_table)
            # 2. Iterate through each catalog dictionary in your primed_tables list
            for i, table_dict in enumerate(primed_tables):
                catalog_name = table_dict['name']  # E.g., 'FERMI', 'VLASS', 'WISE'
                orig_table = table_dict['original_table']

                # Secondary catalogs (VLASS, WISE, etc.) use their name columns
                if catalog_name in result.colnames:
                    matched_indices = result[catalog_name]
                else:
                    print(f"Warning: Catalog column {catalog_name} not found in nway output. Skipping.")
                    continue
        
                # --- HANDLE THE -1 NON-MATCHES SAFELY ---
                # 1. Convert matched_indices to a standard numpy array
                idx_array = np.array(matched_indices)
                safe_indices = np.where(idx_array == -1, 0, idx_array)
                matched_rows = Table(orig_table[safe_indices], masked=True)
                
                # 4. Find exactly which rows were supposed to be non-matches (-1)
                missing_mask = (idx_array == -1)
                
                # 5. Completely mask out every column for those specific rows
                if np.any(missing_mask):
                    for col in matched_rows.colnames:
                        matched_rows[col].mask[missing_mask] = True

                        
                # --- RENAME COLUMNS TO PREVENT DUPLICATES ---
                for col in matched_rows.colnames:
                    matched_rows.rename_column(col, f"{catalog_name}_{col}")
                    
                final_tables_to_combine.append(matched_rows)
            

            # 3. Stack the nway probabilities + every single original column side-by-side
            FITS_TABLE = hstack(final_tables_to_combine)
            # 4. Save to FITS (Masked arrays automatically turn into FITS null values/NaNs)
            # Safe conversion now that result is guaranteed to exist
            # so it stacks them all  like nway does!!
            catalog_path= catalog_reduced_dir + f'{match_type_path}/'+ f'{catalog_year}/'+ f'{source_name}_table_{len(FITS_TABLE)-1}_matches.fits'  

            # # the absolute probability of a source is useless!
            # FITS_TABLE['p_absolute'] = FITS_TABLE['prob_this_match']*FITS_TABLE['prob_has_match']
            threshold = 1000000
            if len(FITS_TABLE) > threshold:
                print(f'{len(FITS_TABLE)} exceeds writing threshold {threshold}, reduce to top X and bottom X matches')
                
                # Clean duplicates to get reliable rank partitions
                p_any_list = list(set(FITS_TABLE['prob_has_match']))
                # 1. Get the fifth HIGHEST threshold (5th from the end)
                fifth_highest_p_any = np.partition(p_any_list, -5)[-5]
                # 2. Get the fifth LOWEST threshold (5th from the beginning)
                fifth_lowest_p_any = np.partition(p_any_list, 4)[4]
                
                print(f"Top cut threshold: {fifth_highest_p_any}")
                print(f"Bottom cut threshold: {fifth_lowest_p_any}")
                
                # 3. Combine both masks using a bitwise OR (|)
                top_mask = FITS_TABLE['prob_has_match'] >= fifth_highest_p_any
                bottom_mask = FITS_TABLE['prob_has_match'] <= fifth_lowest_p_any
                
                FITS_TABLE = FITS_TABLE[top_mask | bottom_mask]
            
            print(f'new matched size {len(FITS_TABLE)}')
                        
    
            if keep_catalogs:
                FITS_TABLE.write(catalog_path, format='fits', overwrite=False)
                print(f'{source_index+1+skip_num}/{catalog_size}')
            else:
                FITS_TABLE.write(catalog_path, format='fits', overwrite=True)
                print(f'{source_index+1+skip_num}/{catalog_size}')
    hdu_table.close()
 
if __name__ == '__main__':
    # run the main file and output all of the source names aswell
    main(terminal_call=False,match_type='mag-prior', keep_catalogs=False)
    # can include an option to add or not add the catalogs to get written?
    # runs nway for all of the parameters 
    # end()