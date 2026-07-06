# import itertools
from pprint import pprint
import glob
from copy import copy
import re

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

# Data handling
import numpy as np

def main():    
    warnings.filterwarnings('ignore', category=AstropyWarning)
    warnings.simplefilter('ignore', category=UnitsWarning)
    # catalog call
    # define home dir
    '''CATALOG CALLS'''
    catalog_home_dir = '/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/'
    catalog_reduced_dir = '/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/reduced_catalogs/'
    catalog_mals_all = '/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/mals_all_bands.fits'
    catalog_names = ['MALS_L', 'SPICE-RACS', 'TGSS', 'RACS_low','RACS_low']

    # catalog_names = ['MALS_L']
    '''FERMI CATALOG CREATION'''
    # not spectral combined
    mals_all_sources= QTable.read(catalog_mals_all, format='fits')
    'Fermi Catalogs'
    fermi_lat_surveys = glob.glob('/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/gll**.fit')
    print(fermi_lat_surveys)
    catalog_years = [8, 10, 12, 14, 16]
    # iniate three catalogs_types for mals regular, uss, and css 
    # could think of just getting the matches in the regular mals and then apply the uss and css limits to but I dont want to yet.
    
    mals_catalog = Catalog(mals_all_sources, ['RAJ2000', 'DEJ2000', 'maj_restoring_beam', 'min_restoring_beam', 'pa_restoring_beam', 'total_flux', 'peak_flux', 'spectral_index_spwfit', 'spectral_index_spwfit_e',  's_code', 'spw_id/WB'] , catalog_type='radio')
    mals_catalog.flux_ratio()
    regular_mals_catalog = copy(mals_catalog)
    # works either way I just did not want to iniate the class twice more 
    # racs
    racs_galatic = QTable.read(catalog_home_dir+'racs_low_galatic_region.fits', format='fits')
    racs_galatic = Catalog(racs_galatic, ['RAJ2000', 'DEJ2000', 'maj', 'min', 'pa', 'total_flux', 'peak_flux', 'N/A', 'N/A',  's_code']
                           , catalog_type='radio')
    racs_galatic.flux_ratio()
    racs_galatic_cut = QTable.read(catalog_home_dir+'racs_low_galatic_cut.fits', format='fits')
    racs_galatic_cut = Catalog(racs_galatic_cut, ['RAJ2000', 'DEJ2000', 'maj', 'min', 'pa', 'total_flux', 'peak_flux', 'N/A', 'N/A',  's_code']
                               , catalog_type='radio')
    racs_galatic_cut.flux_ratio()
    
    
    # I reference the flux and spec index in in the spectral window I; spice-racs
    # the maj and min are of the source not the restoring beam! but it doesnt matter as no source is getting compared to it right now!
    spice_rac_catalog = QTable.read(catalog_home_dir+'spice-racs_catalog.fits', format='fits')
    spice_rac_catalog.remove_column('source_id')
    spice_rac_catalog.remove_column('spectral_index')
    spice_racs_reducer = Catalog(spice_rac_catalog, ['RAJ2000', 'DEJ2000', 'maj_axis', 'min_axis', 'pa', 'total_I_flux', 'peak_I_flux', 'N/A', 'N/A',  's_code']
                                 ,catalog_type='radio')
    spice_racs_reducer.flux_ratio()
    # tgss
    
    # putting in dummy columns for spectral index and err another reason why I need to remove that
    tgss_catalog = QTable.read(catalog_home_dir+'tgss_catalog.fits', format='fits')
    # tgss_catalog.add_column
    tgss_reducer = Catalog(tgss_catalog, ['RAJ2000', 'DEJ2000', 'Maj', 'Min', 'PA', 'Stotal', 'Speak', 'Speak', 'Speak', 'Code']
                           , catalog_type='radio')
    tgss_reducer.flux_ratio()
    
    # this would possible change due to use getting the spectral index from the full band measurements 
    # css_mals_catalog     = copy(mals_catalog.radio_reduction(catalog_type='css'))
    # uss_mals_catalog     = copy(mals_catalog.radio_reduction(catalog_type='uss'))
    
    folder_surveys = ['8_yr/', '10_yr/', '12_yr/', '14_yr/', '16_yr/']
    for cat_year_idx, survey in enumerate(fermi_lat_surveys):
        # fermi call
        survey_hud = fits.open(survey)
        Fermi_catalog = QTable.read(survey_hud, format='fits')
        print(f'====Analysis on {catalog_years[cat_year_idx]}-Yr-Fermi Catalog====')
        # make sure fermi is only calling unassociated sources
        # two step unknown sources only!
        class1_mask = np.isin(Fermi_catalog['CLASS1'], ['unk',''])
        Fermi_catalog = Fermi_catalog[class1_mask]

        print(f'Fermi Unassociated Sources: {len(Fermi_catalog)}')
        fermi_name = 'FERMI'
        fermi_cataloging = Catalog(Fermi_catalog,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'],catalog_name=fermi_name, catalog_type='xray' )
        '''FERMI OVERLAY- MALS, RACS, TGSS'''
        # since this is forced to be in a [] maybe fix it for single iteration if the user decides not to put it into a ls format?
        # fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_cataloging, comparision_catalogs=[regular_mals_catalog, spice_racs_reducer, tgss_reducer,racs_galatic, racs_galatic_cut])
        fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_cataloging, comparision_catalogs=[regular_mals_catalog])

        # fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_cataloging, comparision_catalogs=[regular_mals_catalog, uss_mals_catalog, css_mals_catalog])
        fermi_mals_overlay.is_in_ellipse_V2()
        print('fermi overlay done')
        fermi_ellipses_final, matched_comparision_catalogs, dict_comparisions_fermi, fermi_catalog, full_candidate_catalogs  = fermi_mals_overlay.matched_catalogs_V2(return_parent_catalogs=True)
        
        print(f'size of mal matches:{len(matched_comparision_catalogs[0])}')
        print(f'size of spice-rac matches:{len(matched_comparision_catalogs[1])}')
        print(f'size of tgss matches:{len(matched_comparision_catalogs[2])}')
        print(f'size of rac-low matches:{len(matched_comparision_catalogs[3])+len(matched_comparision_catalogs[4])}')


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
        
        all_checks = False
        matched_spws = []
        # loop each spectral window starting from the lowest for source matches
        while all_checks == False:
            # applying spectral window matching 
            # while all_checks == False:
            # first iteration
            spw_catalogs = {}
            spw_catalogs[0] = mal_cat[mal_cat['spw_id'] ==all_spw_labels[0]]
        
            mask_base_sources = np.isin(mal_cat['Indexes'], spw_catalogs[0]['Indexes'])
            spw_catalogs[1] = mal_cat[~mask_base_sources]
            spw_catalog_class_instances = {}
            
            for i in range(len(spw_catalogs.keys())):
                spw_catalog_class_instances[i] = Catalog(spw_catalogs[i], ['RAJ2000', 'DEJ2000', 'maj_restoring_beam', 'min_restoring_beam', 'pa_restoring_beam', 'total_flux', 'peak_flux', 'spectral_index_spwfit', 'spectral_index_spwfit_e',  's_code', 'spw_id/WB'])
                
            spw__all = CatalogOverlayer(base_catalog=spw_catalog_class_instances[0], comparision_catalogs=[spw_catalog_class_instances[1]])
            spw__all.is_in_ellipse_V2()
                        
            print('Spectral Indexing Complete')
        
            spw_base_matched_catalog, spw_comparision_matched_catalog, dict_spws, spw_base_full_cat, spw_comparision_full_cats = spw__all.matched_catalogs_V2(return_parent_catalogs=True)
        
        # for the single loop of all spw windows with just the matching names
        #    get the global indexes for the dict and the key 
        #  matched_comparision_catalogs[matching_source][global_value_indexes] = matched_comparision_catalogs['Source_Name'][global_key_indexes]
        
            mals_spectral_combined, no_base_match_sources = spectral_window_merging_V2(spw_base_full_cat, spw_comparision_full_cats[0], dict_spws[0])
            # data of current loop
            
            print(f'All matches : {all_spw_labels[0], len(mals_spectral_combined)}')
            # print(mals_spectral_combined)
            print('=====')
            print(f'unassociated sources: {len(no_base_match_sources)}')
            print(f'Cross Spectral Window Matches {len(dict_spws[0].keys())}')
            # print('Spectral windows merge 0 and all complete')    
            
            # prepare for next loop by iterating through all 
            matched_spws.append(mals_spectral_combined)
            mal_cat = no_base_match_sources
            all_spw_labels = all_spw_labels[1:]
            
            if len(all_spw_labels) < 2:
                all_checks=True
                matched_spws.append(mal_cat)
            else:
                # add the unassociated sources as well!
                continue
        mals_all_spws_merged = vstack(matched_spws)
        print(f'all matches {len(mals_all_spws_merged)}')
        
        '''I WOULD HAVE TO CALCULATE THE MALS SOURCES ALPHA VALUES HERE?!? Or do it after! the full generation of the catalogs'''
        # astropy fitting;chi-least-squared-analysis
        # append the slope as the spectral index and the chi squared as the error? per mals point 
        # option plot the sources in this case only!
        model = models.Polynomial1D(degree=1)
        fitter=fitting.LinearLSQFitter()
        freq_pattern     = r'Freq_LSPW*'
        flux_pattern     = r'Flux_LSPW*'
        flux_err_pattern = r'Flux_Error_*'

        mals_all_spws_merged = mals_all_spws_merged.filled(0)
        best_fit_slopes = []
        best_fit_intercepts = []
        chi_values = [] 
        for mals_source in mals_all_spws_merged:
            # make all the (x) freqs in order and (y) fluxes in order
            # reg ex call the freq names Freq_LSPW_
            freq_names = [string for string in mals_source.colnames if re.search(freq_pattern, string)]
            flux_names = [string for string in mals_source.colnames if re.search(flux_pattern, string)]
            flux_err_names = [string for string in mals_source.colnames if re.search(flux_err_pattern, string)]

    
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
        
        '''TEMP JUST FOR MATCHING UNSEP MALS!'''
        mals_all_spws_merged = mal_cat
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
            updated_comparision_catalogs = [mals_all_spws_merged]
            
        else:
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
            
        '''Output the matched RACS, MALS tables, full grouped table, and hdu lists of individual fermi ellipses'''
        # racs_tables 
        # mals_tables
        # full grouped tables
        # individual fermi ellipses
        '''HDU LIST CREATION '''
        
        # create all QTables with the source, 
        hdu_list_holder = [fits.PrimaryHDU()]
        # call local index
        i = 1
        for ellipse_index in range(len(fermi_catalog)):
            print(f'{i}/{len(fermi_catalog)}')
            values =[global_dict.get(ellipse_index,None) for global_dict in global_dict_indexes]
            none_check = all(item == None for item in values)
            # if none just make a unassociated catalog with nothing 
            if none_check:
                # add a column with the ellipse!
                ellipse_table = Table(fermi_catalog[ellipse_index:ellipse_index+1])
                hdu = fits.BinTableHDU(ellipse_table, name=f'{ellipse_table[0]['Source_Name']}')
                hdu_list_holder.append(hdu)
                i+=1
                continue
            else:
                ellipse_table = Table(fermi_catalog[ellipse_index:ellipse_index+1])
                name = ellipse_table[0]['Source_Name']
                # add a column with the ellipse!
                # changed to np.isin -6/26, iterated 6/29
                table_masks = [np.isin(updated_comparision_catalog['Indexes'], value)
                               for updated_comparision_catalog, value in zip(updated_comparision_catalogs,values)]
                catalog_properties = [Table(updated_comparision_catalog[table_mask].filled(0))
                                      for updated_comparision_catalog, table_mask in zip(updated_comparision_catalogs,table_masks)]
                
                ellipse_table = vstack([ellipse_table, *catalog_properties])
                print(f'{len(ellipse_table)} sources')
                # put a value that differentiates unassociated versus associated!
                hdu = fits.BinTableHDU(ellipse_table, name=name)
                hdu_list_holder.append(hdu)
                i+=1
                continue
        hdu_master = fits.HDUList(hdu_list_holder)
        print(len(hdu_master))
        hdu_master.writeto(catalog_reduced_dir + folder_surveys[cat_year_idx]+'TMR_fermi_tables.fits', overwrite=True)
        print(f'{folder_surveys[cat_year_idx]}yr-fermi catalogs complete')

if __name__ == '__main__':
    # print(timeit.timeit('main()', number=1))
    main()
