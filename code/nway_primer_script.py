# takes input ellipse(s) from hdulist
import glob
from astropy.io import fits
from astropy.table import vstack
from astropy.table import QTable, Column, MaskedColumn
import os
# goal of this script is just to find the associations between each DR4/5 and ALL of its associates
# in doing so it will run the fermi overlay script

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.io import fits
from astropy.table import vstack
# from astropy.table import vstack

# Access astronomical databases
from astroquery.vizier import Vizier
from astropy.io.votable import parse
# Data handling
import numpy as np

# reads it
# could be bad but just a basic uncertainity for the catalogs
# change to just ra uncertainity  if necessary!
def pos_unc(catalog):
    match list(set(catalog['catalog_names'])):
        case 'Fermi':
            ra_err = (catalog['Conf_95_SemiMajor']*u.deg).to(u.arcsec)
            dec_err = (catalog['Conf_95_SemiMinor']*u.deg).to(u.arcsec)
            pa = (catalog['FERMI_Conf_95_PosAng']*u.deg).to(u.arcsec)
            
        case 'TGSS'|'MALS':
            ra_err = catalog['ra_mean_e']
            dec_err = catalog['dec_mean_e']
            pa = catalog['pa_e']
        
        # vizier catalog search
        case _:
            ...

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

def main():
    catalog_mals_call = '/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/reduced_catalogs/'
    fermi_list_combined = catalog_mals_call + '8_yr/' + 'fermi_tables.fits'
    # maybe make a easy way to click and choose from a list!
    source_name = '4FGL J1048.4-5030'
    hdu_table = fits.open(fermi_list_combined,  memmap=False)
    # print(hdu_table.info())
    ellipse = QTable(hdu_table[source_name].data)
    # ellipse.remove_column('DEC_Counterpart')
    hdu_table.close()
    
    ellipse.rename_column('DEJ2000', 'DECJ200')

    # vizier call for other sources here
    # check that the ra and dec of other catalogs is in degs 
    # seperates sources based on catalogs and removes it from the fermi ellipse
    # the first catalog should always be the id?

    
    catalogs = []
    catalog_names = list(set(ellipse['catalog_names']))
    for name in catalog_names:
        table_mask = np.isin(ellipse['catalog_names'], name)
        catalog = ellipse[table_mask]
        
        # get the positional uncertanity for each catalog 
        positional_uncer = pos_unc(catalog)
        catalog['pos_err'] = positional_uncer
            
        if catalog['Source_Name'] == source_name:
            catalog['ID'] = 1
            semi_major = catalog['Conf_95_SemiMajor'][0]
            semi_minor = catalog['Conf_95_SemiMinor'][0]
        catalogs.append(catalog)
    
    # print(catalogs)
    # they will all match the same area!
    print(f'search radius (arcsec): {(semi_major*u.deg).to(u.arcsec)}')
    area = np.pi * semi_major * semi_minor
    # makes them all into fits files
    for catalog in catalogs:
        name = list(set(catalog['catalog_names']))
        catalog.meta['EXTNAME'] = name[0]
        catalog.meta['SKYAREA'] = area
        catalog.write(catalog_mals_call + 'temp_nway_files/'+ f'{name[0]}_catalog.fits', format='fits', overwrite='True')
        
# second main def 
# deletes all the sub files unless specified not to
if __name__ == '__main__':
    main()
    # at this point just get the start and then do the nway run manually
    # nway()
    # end()
