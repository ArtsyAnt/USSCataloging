# goal
# find the unassociated to associated sources for ALL FERMI ELLIPSES, and make that into a table @ the 14 year mark
# he wants me to find the 

# final catalog structure 
# [fermi ellipse name, what survey was it found, what survey it became associated in, the values in the original survey, the values in the found survey]
import glob
from astropy.io import fits
from astropy.table import vstack
from astropy.table import QTable, Table, Column, MaskedColumn
import os

from multiwavecataloging import Catalog
from multiwavecataloging import CatalogOverlayer
from multiwavecataloging import fermi_plot
from multiwavecataloging import global_index

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.io import fits
from astropy.table import hstack, vstack
import time 
# from astropy.table import vstack

# Access astronomical databases
from astroquery.vizier import Vizier
# Data handling
import numpy as np

def rename_table_columns(row, prefix):
    """Renames all columns in a table with a prefix to prevent hstack collisions."""
    renamed = Table(row)
    for col in renamed.colnames:
        renamed.rename_column(col, f"{prefix}_{col}")
    return renamed

def main():
    catalog_home_dir = '/lustre/aoc/observers/nm-16041/antonio/USSCataloging/catalogs/'
    catalog_reduced_dir = '/lustre/aoc/observers/nm-16041/antonio/USSCataloging/reduced_catalogs/'
    
    fermi_lat_surveys = glob.glob(catalog_home_dir +'gll**.fit')
    fermi_lat_surveys.sort(reverse=False)
    print(fermi_lat_surveys)
    yr_16 = fermi_lat_surveys.pop()
    # print(yr_16)
    catalog_years = ['08', '10', '12', '14']

    # removed 16 year!cl
    finalized_fermi_mals_dicts = []
    
    # fermi call
    # '''double check it deosnt call 16 year!'''
    
    all_fermi_catalogs_instances = []
    class_prepped_fermi_catalog = []
    Fermi_catalog_base_list = []

    for iteration, survey in enumerate(fermi_lat_surveys):
        survey_hud = fits.open(survey)
        Fermi_catalog = QTable.read(survey_hud[1], format='fits')
        
        Fermi_catalog_Class = Catalog(Fermi_catalog,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'],catalog_name='FERMI_comp', catalog_type='xray')
        all_fermi_catalogs_instances.append(Fermi_catalog_Class)
            
    # Extract unique classes from the 'CLASS1' column
    classes = list(set(Fermi_catalog['CLASS1']))
 
    # Standardize unknown labels (stripping whitespace handles 'unk ' and 'UNK ' automatically)
    association_unknown = ['unk', 'UNK', '',]
    # Filter out the unknown classes to keep only known associations
    association_known = [
        cls for cls in classes 
        if str(cls) not in association_unknown
    ]

    # loop for each catalog 
    for starting_catalog in range(len(catalog_years)):
        Data_release_origin = starting_catalog
        unassoc_assoc_catalog = []

        print(f'====Analysis on {catalog_years[starting_catalog]}-Yr-Fermi Catalog====')
        '''data release call'''
        try:
            # 8 year catalog is special wouldnt have a data release call so try/except
            # only call of the datarelease for candidates not for solutions!
            Fermi_catalog_base = all_fermi_catalogs_instances[starting_catalog].catalog
            Fermi_catalog_base = Fermi_catalog_base[Fermi_catalog_base['DataRelease'] == Data_release_origin+1]
        except:
            # base catalog for 8 year since it wont have a datarelease column
            Fermi_catalog_base = all_fermi_catalogs_instances[starting_catalog].catalog
        # print(Fermi_catalog_base)
            # taken directly

        # for catalog year loop through until it gets it first first match with a class1 association
        class1_mask = np.isin(Fermi_catalog_base['CLASS1'], association_unknown)
        Fermi_catalog_base = Fermi_catalog_base[class1_mask]

        # generate empty index of other cats, and their datarelease 
        # loop through all the sources individually 
        Data_release_comparision = Data_release_origin +1

        for catalog_instance in all_fermi_catalogs_instances[Data_release_comparision:]:
            Fermi_catalog_base[f'Comparision_Cat_Index_{catalog_years[Data_release_comparision]}'] = [np.nan] * len(Fermi_catalog_base)
            Fermi_catalog_base[f'DataRelease_{catalog_years[Data_release_comparision]}']  = [np.nan] * len(Fermi_catalog_base)
            Fermi_catalog_base[f'CLASS1_{catalog_years[Data_release_comparision]}']  = ['   '] * len(Fermi_catalog_base)
            Fermi_catalog_base[f'Source_Name_{catalog_years[Data_release_comparision]}']  = ['                  '] * len(Fermi_catalog_base)
            
 
            
            fermi_reducer = Catalog(Fermi_catalog_base,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'],catalog_name='FERMI_base', catalog_type='xray')
            fermi_overlayer= CatalogOverlayer(base_catalog=fermi_reducer, comparision_catalogs=[catalog_instance])
            try:
                fermi_overlayer.is_in_ellipse_V2()
            except Exception as e:
                print(e)
                continue
            print(f'datarelease: {Data_release_comparision}')
            fermi_ellipses_final, matched_comparision_catalogs, dict_comparisions_fermi, fermi_catalog, full_candidate_catalogs = fermi_overlayer.matched_catalogs_V2(return_parent_catalogs=True)

            class_exists_mask = np.isin(matched_comparision_catalogs[0]['CLASS1'], association_known)
            fermi_catalog = fermi_catalog[0]
            comparision_cat = full_candidate_catalogs[0]
            associated_keys = [key for key, value in dict_comparisions_fermi[0].items() if comparision_cat['CLASS1'][value] in association_known]

            # table 
            for key in associated_keys:
                Fermi_catalog_base[key] [f'DataRelease_{catalog_years[Data_release_comparision]}'] =   comparision_cat[dict_comparisions_fermi[0][key]]['DataRelease'][0]
                Fermi_catalog_base[key][f'Comparision_Cat_Index_{catalog_years[Data_release_comparision]}'] =  dict_comparisions_fermi[0][key][0]
                Fermi_catalog_base[key][f'CLASS1_{catalog_years[Data_release_comparision]}']  = comparision_cat[dict_comparisions_fermi[0][key]]['CLASS1'][0]
                Fermi_catalog_base[key][f'Source_Name_{catalog_years[Data_release_comparision]}']  = comparision_cat[dict_comparisions_fermi[0][key]]['Source_Name'][0]

            Data_release_comparision +=1
            print(f'Associations: {len(associated_keys)}')
        print(Fermi_catalog_base[-8:])
        Fermi_catalog_base_list.append(Fermi_catalog_base)

    # write table!
    for i in range(len(catalog_years)):
        Fermi_catalog_base_list[i].write(catalog_reduced_dir+ 'unass_to_assoc_' + catalog_years[i] +'_yr_matches.fits', format = 'fits', overwrite=True)
        # reset after this!

if __name__ == '__main__':
    main()
