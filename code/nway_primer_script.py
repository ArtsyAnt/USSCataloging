# takes input ellipse(s) from hdulist
import glob
from astropy.io import fits
from astropy.table import vstack
from astropy.table import QTable, Column, MaskedColumn
from astropy.table import Table



import os
# goal of this script is just to find the associations between each DR4/5 and ALL of its associates
# in doing so it will run the fermi overlay script

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import vstack
# from astropy.table import vstack

# Access astronomical databases

from astroquery.vizier import Vizier
from astropy.io.votable import parse
# Data handling
import numpy as np
# import pymc as pm
from nwaylib import nway_match
from astropy.coordinates import ICRS, FK5


# ref from nway-apitest.py
def table_from_fits(fitsname, poserr_value=None, area=None, magnitude_columns=[]):
	fits_table = fits.open(fitsname)[1]
	table_name = fits_table.name
	ra = fits_table.data['RA']
	dec = fits_table.data['DEC']
	if 'pos_err' in fits_table.data.columns.names:
		poserr = fits_table.data['pos_err']
	else:
		assert poserr_value is not None, ('"pos_err" column not found in file "%s", and no poserr_value passed' % fitsname)
		poserr = poserr_value * np.ones(len(ra))
	if area is None:
		area = fits_table.header['SKYAREA'] * 1.0
  
	# magnitude columns
	mags = []
	maghists = []
	magnames = []
	#for mag in magnitude_columns:
	for col_name, magfile in magnitude_columns:
		assert col_name in fits_table.data.dtype.names

		mag_all = fits_table.data[col_name]
		# mark -99 as undefined
		mag_all[mag_all == -99] = np.nan

		mags.append(mag_all)
		magnames.append(col_name)
		if magfile == 'auto':
			maghists.append(None)
		else:
			bins_lo, bins_hi, hist_sel, hist_all = np.loadtxt(magfile).transpose()
			maghists.append((bins_lo, bins_hi, hist_sel, hist_all))

	return dict(name=table_name, ra=ra, dec=dec, error=poserr, area=area, mags=mags, maghists=maghists, magnames=magnames)
	# area in square degrees
	# error in arcsec
	# ra/dec in degrees
	# mag: column of something
	# maghists: either (bin, sel, all) tuple or None (for auto)

def table_from_catalog(catalog, name, poserr_value=None, area=None, magnitude_columns=[]):
	fits_table = catalog
	table_name = name
	ra = fits_table['RA']
	dec = fits_table['DEC']
	poserr = fits_table['pos_err']

	if area is None:
		area = skyarea * 1.0
  
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
		else:
			bins_lo, bins_hi, hist_sel, hist_all = np.loadtxt(magfile).transpose()
			maghists.append((bins_lo, bins_hi, hist_sel, hist_all))

	return dict(name=table_name, ra=ra, dec=dec, error=poserr, area=area, mags=mags, maghists=maghists, magnames=magnames)
	# area in square degrees
	# error in arcsec
	# ra/dec in degrees
	# mag: column of something
	# maghists: either (bin, sel, all) tuple or None (for auto)
 
 
# creates an offset for the fermi ellipses
def create_fake_catalog(catalog):
    # make a random ra and dec offset that is X greater than the 95% positional uncertainities! 
    # call position parameters
    ra = (catalog['RA'][0])
    dec = (catalog['DEC'][0])
    ra_err = (catalog['Conf_95_SemiMajor'][0])
    dec_err = (catalog['Conf_95_SemiMinor'][0])

    # generate random coordinates
    # randomize the size of offset
    # wont be in the inner half of the ra and dec err 
    random_ra  = np.random.uniform((ra+ra_err*0.5), (ra+ra_err))
    print(random_ra)
    random_dec = np.random.uniform((dec+dec_err*0.5), (dec+dec_err))
    print(random_dec)

    # update coordinates
    fake_catalog = catalog.copy()
    fake_catalog['RA'] = random_ra
    fake_catalog['DEC'] = random_dec
    fake_catalog['catalog_names'] = 'FAKE_' +fake_catalog['catalog_names']
    return fake_catalog
    # return new catalog 
    
# could be bad but just a basic uncertainity for the catalogs
# change to just ra uncertainity if necessary!
def pos_unc(catalog):
    match catalog['catalog_names'][0]:
        case 'FERMI':
            ra_err = (catalog['Conf_95_SemiMajor']*u.deg).to(u.arcsec)
            dec_err = (catalog['Conf_95_SemiMinor']*u.deg).to(u.arcsec)
            pa = (catalog['Conf_95_PosAng']*u.deg)

        case 'TGSS'|'MALS_L'|'RACS_low':
            ra_err = catalog['ra_mean_e']
            dec_err = catalog['dec_mean_e']
            pa = catalog['pa_e']*u.deg

        case 'WISE':
            ra_err = catalog['eeMaj']
            dec_err = catalog['eeMin']
            pa = catalog['eePA']

        case 'PANSTARRS':
            ra_err = catalog['e_RAJ2000']
            dec_err = catalog['e_DEJ2000']
            pa = 0

        case 'GAIA':
            ra_err = catalog['RA_e']
            dec_err = catalog['DEC_e']
            pa = 0
            
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
def main(catalog_year ='14', terminal_call =True, offset_calibration=False, keep_catalogs=True, fermi_names=None, overwrite=False):
    catalog_mals_call = '/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/reduced_catalogs/'
    fermi_list_combined = catalog_mals_call + catalog_year +'_yr/' + 'MALS_spws_fermi_tables.fits'
    # maybe make a easy way to click and choose from a list!
    
    hdu_table = fits.open(fermi_list_combined,  memmap=False)
    # source_name = '4FGL J0041.3-0048'
    source_names = [hdu.name for hdu in hdu_table]
    source_names = source_names[1:]
    catalog_size = len(source_names)
    
    # only look for sources that I havent already dont
    files_to_skip =  glob.glob(catalog_mals_call + 'all_nway_matches/fakes/'+ '4FGL**')
    files_to_skip  = [source.split('_')[0] for source in files_to_skip]
    skip_num = len(files_to_skip)
    print(files_to_skip)
    
    for source_index, source_name in enumerate(source_names):
        print(f'======{source_name}======')
        ellipse = QTable(hdu_table[source_name].data)
        # ellipse.remove_column('DEC_Counterpart')
        # best to rename all files to work with the nway internal and extrenal call!
        ellipse.rename_column('RAJ2000', 'RA')
        ellipse.rename_column('DEJ2000', 'DEC')

        catalogs = []
        catalog_names = list(set(ellipse['catalog_names']))
        for name in catalog_names:
            table_mask = np.isin(ellipse['catalog_names'], name)
            catalog = ellipse[table_mask]
        
            # remove the empty columns created when merging the base radio catalogs! add more when necessary
            cols_to_remove = [col_name for col_name in catalog.colnames if np.all(np.isin(catalog[col_name], ['--', '[]', '', ' ']))]
            protected_cols = {'CLASS1', 's_code', 'ASSOC1', 'Source_Name'}
            cols_to_remove = [col for col in cols_to_remove if col not in protected_cols]
            
            print('removing columns')
            print(cols_to_remove)
            catalog.remove_columns(cols_to_remove)

            # get the positional uncertanity for each catalog 
            positional_uncer = pos_unc(catalog)
            catalog['pos_err'] = positional_uncer
                
            # used to identify the fermi catalog I could probably combine both the for loops since they do the same process of iterating between catalogs 
            if catalog['Source_Name'][0] == source_name:
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
    
        print('start vizier search')

        # gaia, wise, and panstarrs call
        overlays = {'I/355': ['GAIA', '*', 'e_RA_ICRS', 'e_DE_ICRS'],
                    'II/311': ['WISE', '*', 'eePA']
                    , 'II/349':['PANSTARRS', '*']}
        
        print(f'Source Overlays Final Numbers:')
        for overlay, column in overlays.items():
            catalog_name = column.pop(0)
            try:
                vizier = Vizier(columns=column)
                vizier.ROW_LIMIT = -1# -1 means unlimited
                overlay_catalog = vizier.query_region(
                                                      coord(ra=ra, dec=dec,
                                                            unit=(u.deg, u.deg),
                                                            frame='fk5'), radius=Angle(semi_minor, "deg"),  catalog=[overlay])[0]
            except:
                continue
            
            match overlay:
                case 'I/355':
                    ra_gaia = overlay_catalog['RA_ICRS']
                    dec_gaia = overlay_catalog['DE_ICRS']
                    ra_gaia_e = overlay_catalog['e_RA_ICRS']
                    dec_gaia_e = overlay_catalog['e_DE_ICRS']

                    c_icrs = SkyCoord(ra=ra_gaia , dec=dec_gaia , frame='icrs')
                    c_icrs_e = SkyCoord(ra=ra_gaia_e.to(u.arcsec), dec=dec_gaia_e.to(u.arcsec), frame='icrs')

                    # Transform to J2000 (which Astropy treats as the FK5 frame)
                    c_j2000 = c_icrs.transform_to(FK5(equinox='J2000.0'))
                    c_j2000_e = c_icrs_e.transform_to(FK5(equinox='J2000.0'))

                    ra_gaia = c_j2000.ra.degree
                    dec_gaia = c_j2000.dec.degree
                    
                    ra_gaia_e = c_j2000_e.ra
                    dec_gaia_e = c_j2000_e.dec

                    overlay_catalog['RA_ICRS'] = ra_gaia
                    overlay_catalog['DEC_ICRS'] = dec_gaia
                    overlay_catalog['e_RA_ICRS'] = ra_gaia_e.to(u.arcsec)
                    overlay_catalog['e_DEC_ICRS'] = dec_gaia_e.to(u.arcsec)
                    
                    overlay_catalog.rename_columns(['RA_ICRS','DEC_ICRS', 'e_RA_ICRS', 'e_DEC_ICRS'],
                                                ['RA', 'DEC', 'RA_e', 'DEC_e'])
                case _:
                    overlay_catalog.rename_columns(['RAJ2000', 'DEJ2000'], ['RA', 'DEC'])
            
            overlay_catalog['catalog_names'] = [catalog_name] * len(overlay_catalog)
            overlay_catalog_pos_uncr = pos_unc(overlay_catalog)
            overlay_catalog['pos_err'] = overlay_catalog_pos_uncr
            
            overlay_catalog = overlay_catalog[overlay_catalog['pos_err'] > 0]
            catalogs.append(overlay_catalog)
            print(f'{catalog_name}: {len(overlay_catalog)}' )

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
            catalog_path= catalog_mals_call + 'send_test/'+ f'{name[0]}_catalog.fits'  
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
            primed_tables =[]
            for catalog in catalogs:
                catalog = catalog.filled(0)
                name = catalog.meta['EXTNAME']
                skyarea = catalog.meta['SKYAREA']
                catalog=Table(catalog.as_array())
                match name:
                    case 'FERMI'|'GAIA'|'PANSTARRS'|'WISE':
                        primed_table = table_from_catalog(catalog, name, area=skyarea)
                        primed_tables.append(primed_table)

                    # all radio catalogs get the total flux inquiry?
                    case 'MALS_L'|'TGSS'|'RACS_low':
                        if len(catalog) <=100:
                            primed_table = table_from_catalog(catalog, name, area=skyarea)
                        else:
                            primed_table = table_from_catalog(catalog, name, area=skyarea, magnitude_columns=[('total_flux', 'auto')])
                        primed_tables.append(primed_table)

                    case 'FAKE_FERMI':
                        catalog_path= catalog_mals_call + 'all_nway_matches/fakes/'+ f'{source_name}_fermi_table.fits'  
                        catalog.write(catalog_path, format='fits', overwrite=True)


            # get mag columns from each 
            # put off till later 
            if len(primed_tables) ==1:
                print('no matches')
                print(f'{source_index+1+skip_num}/{catalog_size}')
                continue
            result = nway_match(
                primed_tables,
                prior_completeness=1,
                match_radius = radius_search.value, # in arcsec
                store_mag_hists = False,
                mag_include_radius = mag_search_radius.value, # in arcsec #check later
                )
            
            FITS_TABLE = Table.from_pandas(result)
            # print(FITS_TABLE.colnames)
            catalog_path= catalog_mals_call + 'all_nway_matches/'+ f'{source_name}_table_{len(FITS_TABLE)-1}.fits'  
            # print(FITS_TABLE['prob_has_match'])
            # print(FITS_TABLE['prob_this_match'])

            # the absolute probability of a source
            FITS_TABLE['p_absolute'] = FITS_TABLE['prob_this_match']*FITS_TABLE['prob_has_match']
            FITS_TABLE.write(catalog_path, format='fits', overwrite=True)
            print(f'{source_index+1+skip_num}/{catalog_size}')
    hdu_table.close()
 
if __name__ == '__main__':
    # run the main file and output all of the source names aswell
    main(terminal_call=False)
    # can include an option to add or not add the catalogs to get written?
    # runs nway for all of the parameters 
    # end()
