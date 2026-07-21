import glob
from astropy.io import fits
from astropy.table import vstack
from astropy.table import QTable, Column, MaskedColumn
import os
# goal of this script is just to find the associations between each DR4/5 and ALL of its associates
# in doing so it will run the fermi overlay script

# example of Fermi Unassociated Sources in the MeerKAT Absorption Line Survey catalog reduction using classes
# this should use the 3rd data release which is v31 not v35
from multiwavecataloging import Catalog
from multiwavecataloging import CatalogOverlayer
from multiwavecataloging import fermi_plot
from multiwavecataloging import spectral_window_merging
from multiwavecataloging import global_index

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.io import fits
from astropy.table import vstack
# from astropy.table import vstack

# Access astronomical databases
from astroquery.vizier import Vizier
# Data handling
import numpy as np



# call x surveys across years 
fermi_lat_surveys = glob.glob('/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/gll**.fit')
catalog_years = [8, 10, 12, 14, 16]
# catalog_starting_input = input('Select Starting Catalog Year \nOptions 8, 10, 12, 14, 16\nType Here: ')

# not call a dict because I dont need to in the future change
finalized_fermi_mals_dicts = []

# variable of which catalog I want to start and end with
# this should be the only variable aspect of this script 
starting_catalog=2
ending_catalog = 5
# the iteration the catalog process is on
iteration = 1
# this is just the nature of indexing vs. the actual data release number
# dont call it in the loop as it would not be consistent
Data_release = starting_catalog +1 

# could define a function where it asks for what catalog you want to see its unassoc source, and it takes that input 
# and converts it to a index for this for loop

# only call this catalog once
# catalog call
vizier = Vizier(columns=[])
vizier.ROW_LIMIT = -1# -1 means unlimited
mals_catalog = vizier.get_catalogs('J/ApJS/270/33')[0]

for survey in fermi_lat_surveys[starting_catalog:ending_catalog]:
    # fermi call
    survey_hud = fits.open(survey)
    Fermi_catalog = QTable(survey_hud[1].data)
    print(f'====Analysis on {catalog_years[starting_catalog]}-Yr-Fermi Catalog====')
    # provides the maybe and unknown associations

    # maybe
    # in future add a takeout for the associated findings to effectively 'ban' those names from further inquiry
    # cause we want to look for the catalog it was found in an how?
    # or just keep and can compare if those labels are changed in future!
    # for the starting catalog only look for unknown sources 
    
    '''data release call'''
    # 8 year catalog is special wouldnt have a data release call
    try:
        Fermi_catalog = Fermi_catalog[Fermi_catalog['DataRelease'] == Data_release]
    except:
        pass
    
    if iteration == 1:
        # you apply the himes example script to get the fermi_source matches 
        # then you would get the fermi keys and get the name of those sources 
        all_associations_types = list(set(Fermi_catalog['CLASS1']))
        remove_from_list = ['unk  ', '     ']
        all_associations_types_reduced = [element for element in all_associations_types if element not in remove_from_list]
        # unknown_associated_sources = [class_type for class_type in all_associations_types if class_type.islower()]
        # unknown_associated_sources.append('')
        # class_type_mask = np.isin(Fermi_catalog['CLASS1'], unknown_associated_sources)
        
        # two step unknown sources only!
        class1_mask = np.isin(Fermi_catalog['CLASS1'], ['unk  ', 'UNK  ', '     '])
        Fermi_catalog = Fermi_catalog[class1_mask]
        class2_mask= np.isin(Fermi_catalog['CLASS2'], ['unk       ', 'UNK       ', '          '])
        Fermi_catalog = Fermi_catalog[class2_mask]
        
        '''I could otherwise change this process to be all unassociated sources in the first iteration then look in the next iterations '''
        '''himes_script_call'''
        '''would like to be able to just call a basic reduction script that could take a fermi lat as a input'''
        mals_reordered = mals_catalog['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit', 'e_Spwfit', 'Scode', 'SPW/WB']
        fermi_reordered = Fermi_catalog['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng', 'Source_Name']

        '''FERMI OVERLAY'''
        # i would call in this into the catalog overlay not anything else
        mals_cataloging_uss = Catalog(mals_reordered, ['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit', 'e_Spwfit',  'Scode', 'SPW/WB'], catalog_type='uss')
        # mals_cataloging_css = Catalog(mals_reordered_2, ['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit'], catalog_type='css')
        fermi_cataloging = Catalog(fermi_reordered,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng', 'Source_Name'], catalog_type='xray' )
        # since this is forced to be in a [] maybe fix it for single iteration if the user decides not to put it into a ls format?
        fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_cataloging, comparision_catalogs=[mals_cataloging_uss])

        fermi_ellipses_final, mal_cat_final, dict_mal_femi, uss_candidate_catalog  = fermi_mals_overlay.matched_catalogs()
        fermi_ellipses_final = fermi_ellipses_final[0]
        mal_cat_final = mal_cat_final[0]
        # ls all catalogs without the matching
        # dict key
        dict_mal_femi = dict_mal_femi[0]
        # dict_values = [*dict_mal_femi.values()]
        # dict_keys = list(dict_mal_femi.keys())
    
        '''SPW CROSSMATCHING'''
        spw2_catalog = mal_cat_final[mal_cat_final['SPW/WB'] == 'LSPW_2']
        spw9_catalog = mal_cat_final[mal_cat_final['SPW/WB'] == 'LSPW_9']

        spw2_cataloging = Catalog(spw2_catalog, ['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit', 'Scode', 'SPW/WB'])
        spw9_cataloging = Catalog(spw9_catalog, ['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit', 'Scode', 'SPW/WB'])
        spw2_spw9_overlay = CatalogOverlayer(base_catalog=spw2_cataloging, comparision_catalogs=[spw9_cataloging], beam_scale=2)

        # this will return only the matched sources while the mal_cat_final wil have the indexes of all sources,
        # so the indexes in mal_cat not in the spw2_9 match are the ones of interest 
        spw2_matched_catalog, spw9_matched_catalog, dict_spw2_sp9, spw9_all_candidates = spw2_spw9_overlay.matched_catalogs()
        spw2_matched_catalog = spw2_matched_catalog[0]
        spw9_matched_catalog = spw9_matched_catalog[0]
        dict_spw2_sp9 = dict_spw2_sp9[0]
        spw9_all_candidates = spw9_all_candidates[0]

        print(f'All Sources {len(mal_cat_final)}')
        print(f'Cross Spectral Window Matches {len(dict_spw2_sp9.keys())}')
        
        # combines nearby sources across spectral bands
        unique_sources_sp2 = spectral_window_merging(spw2_catalog, spw9_catalog, dict_spw2_sp9 )
        dict_mal_femi_global_index = global_index(dict_mal_femi, uss_candidate_catalog)
        dict_mal_femi_updated = {key: [value for value in value_list if value in (unique_sources_sp2['Indexes'])] 
                                for key, value_list in dict_mal_femi_global_index.items()
                                }
        dict_values = [*dict_mal_femi_updated.values()]
        dict_keys = list(dict_mal_femi_updated.keys())
        
        finalized_fermi_mals_dicts.append(dict_keys)
        finalized_fermi_mals_dicts.append(dict_values)
        finalized_fermi_mals_dicts.append(fermi_ellipses_final['Source_Name'])
        
        print(fermi_ellipses_final['Source_Name'])
        print('Hime Script Reduction Complete!')
    else:
        # name call SPLIT THE DATA RELEASE YEAR VERSION WITH THE SOURCE NAME
        unassoc_source_names = finalized_fermi_mals_dicts[2]
        # the way I am doing this only works if my inital catalog is in the 4th data release
        # otherwise I would have to use the association flags for the other years (the 0-4 are 4th data releases)
        match starting_catalog:
            case 0|1|2|3:
                compared_fermi_names = Fermi_catalog['Source_Name']
            case 4:
                # different for 16 year cat because it has associated source names as it changed from the two sets
                compared_fermi_names = Fermi_catalog['ASSOC_FGL']
            
        unassoc_source_names_stripped = []
        compared_fermi_names_stripped = []
        # for loop to split the name of the source to remove the Data Release(DR) and usable across DR's
        for source_name in unassoc_source_names:
            unassoc_source_name = source_name.split()
            unassoc_source_names_stripped.append(unassoc_source_name[1])
            
        for source_name in compared_fermi_names:
            # the assoc call is weird? had to double in?
            compared_source_name = source_name.split()
            try:
                compared_fermi_names_stripped.append(compared_source_name[1])
            except:
                compared_fermi_names_stripped.append('N/A')
                    
        # print(compared_fermi_names_stripped)
        # you look for the matching names in the new catalog release
        name_boolean = np.isin(compared_fermi_names_stripped, unassoc_source_names_stripped)
        Fermi_Catalog_matching_names = Fermi_catalog[name_boolean]
        # show all of those sources in the new catalog
        Fermi_Catalog_matching_names

        # this phase I would add any columns to the new catalog about important data of old cat
            # so if there is a lowercase associated add it, flag add it, and ass type old add it
        
        # check if they have a CONFIRMED associated source 
        # ls comphresion confirmed
        Fermi_confirmed_assoc_index = [index for index in range(len(Fermi_Catalog_matching_names)) 
                                       if np.any(np.isin(Fermi_Catalog_matching_names['CLASS1'][index], all_associations_types_reduced))]
        
        # ls comphresion unconfirmeds
        Fermi_unconfirmed_assoc_index = [index for index in range(len(Fermi_Catalog_matching_names))
                                         if np.any(np.isin(Fermi_Catalog_matching_names['CLASS1'][index], ['unk  ', 'UNK  ', '     ', ]))]
        # Fermi_unconfirmed_assoc_index.append[index for index in range(len(Fermi_Catalog_matching_names))
        #                                  if np.any(np.isin(Fermi_Catalog_matching_names['CLASS2'][index], ['unk  ', 'UNK  ', '     ', ]))]
        
            # if they do add that index to a list
        
        newly_asscoiated   = Fermi_Catalog_matching_names[Fermi_confirmed_assoc_index]
        still_unassociated = Fermi_Catalog_matching_names[Fermi_unconfirmed_assoc_index]
        print(newly_asscoiated)
        print('==============================')
        print(still_unassociated)
        
        print(f'{len(still_unassociated)} unassociated sources out of {len(Fermi_Catalog_matching_names)} MALS matched fermi ellipses')
        # apply a mask and a inverse mask to get CONFIRMED and STILL UNCONFIRMED Ellipses catalog
        print(f'===={catalog_years[starting_catalog]}-Yr-Fermi Catalog Analysis Complete====')    
        
    print('up to this point works!')
    starting_catalog += 1
    iteration += 1

# Fermi_hud = fits.open('/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/gll_psc_v31.fit')
# Fermi_catalog = Table(Fermi_hud[1].data)

# 1. call all fermi-lats surveys iterable

# 2.
# def function to add the unassociated column

# 3.
# def function to run through a catalogs (only doing the sources new in that data release)

# 4.
# get all sources with DR association and merge into one

# 5.
# Add column for name association to the catalog

# 6. 
# Develop a way to search for the associated sources (and


