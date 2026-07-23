# preps any and all catalogs
import glob
from astropy.io import fits
from astropy.table import vstack
from astropy.table import QTable, Column, MaskedColumn
from astropy.table import Table

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.coordinates import Angle
from astropy.table import vstack

# Access astronomical databases
from astroquery.vizier import Vizier
from astropy.io.votable import parse
# Data handling
import numpy as np
from nwaylib import nway_match

from astropy.io import fits
from astropy.table import QTable

# removing bad columns that dont read!
def safe_fits_read(filepath):
    """
    Opens a FITS file, strips invalid TDISP format string properties,
    forces memory-mapping to False to unlock the file handle on disk,
    and returns a clean, fully-independent QTable.
    """
    # Open the file handles explicitly
    hdulist = fits.open(filepath, memmap=False)
    
    try:
        for hdu in hdulist:
            if hdu.is_image is False and hasattr(hdu, 'columns'):
                for col in hdu.columns:
                    if col.disp is not None:
                        col.disp = None
        
        # memmap=False copies all table arrays straight into RAM
        # copy=True prevents references to old structures
        table = QTable.read(hdulist, format='fits', memmap=False)
        return table
        
    finally:
        # Explicitly close the file handle to remove the OS lock immediately
        hdulist.close()

# # Re-run your reading setup with this updated function
# NVSS  = safe_fits_read(catalog_home_dir + 'NVSS.dat.gz.fits')
# VLASS = safe_fits_read(catalog_home_dir + 'VLASS_J_ApJS_255_30_comp.dat.fits')
# SUMSS = safe_fits_read(catalog_home_dir + 'SUMSS_VIII_81B_212.dat.gz.fits')
# TGSS  = safe_fits_read(catalog_home_dir + 'TGSS_catalog.fits')
# FIRST = safe_fits_read(catalog_home_dir + 'FIRST_14dec17.fits')




# uses the name in the freqs to match with the catalogs
def frequency_matcher(catalog_dict, catalogs, catalog_names):
    for index, catalog in enumerate(catalogs):
        name = catalog_names[index]
        freq_value = catalog_dict.get(name) 
        
        freq_array = np.full(len(catalog), freq_value.to(u.Hz).value)
        freq_column = Column(name='ref_freq', data=freq_array, unit=u.Hz)
        
        if 'ref_freq' in catalog.colnames:
            catalog.replace_column('ref_freq', freq_column)
        else:
            catalog.add_column(freq_column)
    return catalogs

def main():
    catalog_home_dir = '/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/'
    master_list = []
    # dont run mals here take too long! amd also local computer cant handle it, though a stronger one likley would
    catalog_names = ['VLASS', 'FIRST', 'NVSS', 'SUMSS', 'TGSS', 'racs_low']
    
    catalogs = []
    all_paths = []
    mals_catalogs = []
    processed_names = [] # Keep track of names for the frequency matcher
    
    for name in catalog_names:
        catalog_str = glob.glob(catalog_home_dir+name+'*.fits')
        print(catalog_str)
        
        for strs in catalog_str:
            all_paths.append(strs)
            try:                 
                table = safe_fits_read(strs)
                if name == 'NVSS':
                    coordinate_strings = []
                    for i in range(len(table)):
                        ra_str = f"{table['RAh'][i]}:{table['RAm'][i]}:{table['RAs'][i]}"
                        dec_str = f"{table['DE-'][i]}{table['DEd'][i]}:{table['DEm'][i]}:{table['DEs'][i]}"
                        coordinate_strings.append(f"{ra_str} {dec_str}")
                    coords = SkyCoord(coordinate_strings, unit=(u.hourangle, u.deg), frame='fk5')
                    ra_deg = coords.ra.degree
                    dec_deg = coords.dec.degree
                    
                    if 'RAJ2000' in table.colnames:
                        table.replace_column('RAJ2000', ra_deg)
                    else:
                        table.add_column(ra_deg, name='RAJ2000')
                        
                    if 'DEJ2000' in table.colnames:
                        table.replace_column('DEJ2000', dec_deg)
                    else:
                        table.add_column(dec_deg, name='DEJ2000')
                    table['RAJ2000'].unit = u.deg
                    table['DEJ2000'].unit = u.deg
                    
                # if name == 'mals_all':
                #     mals_catalogs.append(table)
                else:
                    catalogs.append(table)
                    processed_names.append(name) # Save name for frequency lookups
            except Exception as e:
                print(f"Error processing {strs}: {e}")

    '''RENAMING SOURCES '''
    # add
    
    
    ...
    '''FREQUENCY MATCHING'''
    catalog_freq_mapping = {
        'VLASS': 3.0 * u.GHz,
        'FIRST': 1.4 * u.GHz,
        'NVSS': 1.4 * u.GHz,
        'SUMSS': 843.0 * u.MHz,
        'TGSS': 150.0 * u.MHz,
        'racs_low': 887.5 * u.MHz
    }
    
    catalogs = frequency_matcher(catalog_freq_mapping, catalogs, processed_names)
    final_catalogs = catalogs + mals_catalogs
    
    for index, catalog in enumerate(final_catalogs):
        catalog.write(all_paths[index], format='fits', overwrite=True)
    
if __name__ == '__main__':
    main()
