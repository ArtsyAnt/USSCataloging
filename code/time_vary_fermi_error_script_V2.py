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
    catalog_reduced_dir = '/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/reduced_catalogs/'

    # call x surveys across years 
    fermi_lat_surveys = glob.glob('/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/gll**.fit')
    yr_16 = fermi_lat_surveys.pop()
    print(yr_16)
    catalog_years = ['8', '10', '12', '14']
    # removed 16 year!cl
    finalized_fermi_mals_dicts = []
    unassoc_assoc_all_catalogs = []
    # fermi call
    '''double check it deosnt call 16 year!'''
    all_fermi_catalogs = []
    class_prepped_fermi_catalog = []
    for iteration, survey in enumerate(fermi_lat_surveys):
        survey_hud = fits.open(survey)
        Fermi_catalog = QTable(survey_hud[1].data)
        
        if iteration >=1:
            all_fermi_catalogs.append(Fermi_catalog)
            
        
        
    # loop for each catalog 
    for starting_catalog in range(len(catalog_years)):
        Data_release_origin = starting_catalog
        unassoc_assoc_catalog = []

        print(f'====Analysis on {catalog_years[starting_catalog]}-Yr-Fermi Catalog====')
        '''data release call'''
        try:
            # 8 year catalog is special wouldnt have a data release call so try/except
            # only call of the datarelease for candidates not for solutions!
            Fermi_catalog_base = all_fermi_catalogs[starting_catalog]
            Fermi_catalog_base = Fermi_catalog_base[Fermi_catalog_base['DataRelease'] == Data_release_origin]
        except:
            pass
        # base catalog 
        Fermi_catalog_base = all_fermi_catalogs[starting_catalog]
        # taken directly
        associations_unknown = ['unk  ', 'UNK  ', '     ']
        class1_mask = np.isin(Fermi_catalog_base['CLASS1'], associations_unknown)
        Fermi_catalog_base = Fermi_catalog_base[class1_mask]
        
        fermi_reducer = Catalog(Fermi_catalog_base,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'],catalog_name='FERMI', catalog_type='xray')
        x= CatalogOverlayer(base_catalog=Fermi_catalog_base, comparision_catalogs=[])
        x.is_in_ellipse_V2
        x.matched_catalogs_V2
        # check source name
        # check assoc_flg!
        # compared_fermi_names = Fermi_catalog['Source_Name']
        # compared_fermi_names_assoc_fgl_flag = Fermi_catalog['ASSOC_FGL']
        
        # check if source is in either assoc_fgl or just as its position!
        for source in Fermi_catalog_base:
            unassoc_source_name = source['Source_Name'].split()
            unassoc_source_names_stripped = unassoc_source_name[1]
            associated_row = None 

            # data release changes over each catalog 
            data_release_comparision = starting_catalog+1
            for comp_catalog in all_fermi_catalogs[starting_catalog+1:]:
                # see if the source cneter is in the semi_major axis 
                # assoc matching
                # print(source['Source_Name'])
                # print(comp_catalog['ASSOC_FGL'])
                # THE ASSOC FLAG IS NOT RIGHT YOU FOOL ITS FOR PREVIOUS DATA SETS BEFORE 8 YEARS 
                
                
                compared_source_name = [comp_source['Source_Name'].split()[1] for comp_source in comp_catalog]
                source_mask = np.isin(compared_source_name, unassoc_source_names_stripped)
                if np.any(source_mask):
                    associated_row = comp_catalog[source_mask]
                    break
                
                data_release_comparision+=1

            if associated_row == None:
                continue
            print(data_release_comparision)
            associated_row = rename_table_columns(associated_row, prefix=f'Fermi_{catalog_years[data_release_comparision]}')
            source_row = rename_table_columns(source, prefix=f'Fermi_{catalog_years[Data_release_origin]}')
            data_release_found = [data_release_comparision]
            data_release_begun = [Data_release_origin]
            if associated_row[f'Fermi_{catalog_years[data_release_comparision]}_ASSOC1'] not in associations_unknown:
                association = associated_row[f'Fermi_{catalog_years[data_release_comparision]}_ASSOC1']
                association_transition = hstack([source_row, associated_row, data_release_begun, data_release_found, association])
                unassoc_assoc_catalog.append(association_transition)
                print(len(unassoc_assoc_catalog))
        unassoc_Table = Table(unassoc_assoc_catalog)
        unassoc_Table.meta['YEAR'] = catalog_years[starting_catalog]
        unassoc_Table.write(catalog_reduced_dir+ 'unass_to_assoc' + catalog_years[starting_catalog] +'_yr_matches.fits', format = 'fits')
        unassoc_assoc_all_catalogs.append(unassoc_assoc_catalog)
        # reset after this!

if __name__ == '__main__':
    main()