# preps any and all catalogs
import glob
import os

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
def safe_fits_read(filepath, name):
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
                        
        if name == 'SUMSS':
            try:
                table = QTable.read(hdulist, format='fits', memmap=False)
                return table
            # exception is for first run where the bad column is there!
            except Exception as e:
                print(e)
                # Grab the targeted column name using index 18
                col_to_remove = hdulist[1].columns.names[18]
                print(f'\nRemoving column: {col_to_remove}\n')
                # Filter out the unwanted column definition
                remaining_cols = [c for c in hdu.columns if c.name != col_to_remove]
                
                # Rebuild the column definitions and create a fresh TableHDU
                new_cols = fits.ColDefs(remaining_cols)
                new_hdu = fits.TableHDU.from_columns(new_cols, header=hdu.header)
                table = QTable.read(new_hdu, format='fits', memmap=False)
                return table

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
    current_working_dir = os.getcwd()
    catalog_home_dir = current_working_dir + '/catalogs/'
    
    master_list = []
    # dont run mals here take too long! amd also local computer cant handle it, though a stronger one likley would
    # catalog_names = ['VLASS', 'FIRST', 'NVSS', 'SUMSS', 'TGSS', 'racs_low']
    catalog_names = ['NVSS', 'SUMSS']
    
    catalogs = []
    all_paths = []
    mals_catalogs = []
    processed_names = [] # Keep track of names for the frequency matcher
    refitting_list = ['NVSS', 'SUMSS']
    for name in catalog_names:
        catalog_str = glob.glob(catalog_home_dir+name+'*.fits')
        print(catalog_str)
        
        for strs in catalog_str:
            all_paths.append(strs)
            try:                 
                table = safe_fits_read(strs, name)
                if name in refitting_list:
                    coordinate_strings = []
                    try:
                        # 1. Convert columns directly to standard numpy arrays (No string chopping!)
                        rah = np.asarray(table['RAh'], dtype=float)
                        ram = np.asarray(table['RAm'], dtype=float)
                        ras = np.asarray(table['RAs'], dtype=float)
                        
                        ded = np.asarray(table['DEd'], dtype=float)
                        dem = np.asarray(table['DEm'], dtype=float)
                        des = np.asarray(table['DEs'], dtype=float)
                        
                        # 2. Extract the sign correctly without string errors
                        if 'DE-' in table.colnames:
                            # If it's bytes, decode it; handle array masks cleanly
                            signs = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in table['DE-']])
                            # Map '-' to -1.0, everything else to +1.0
                            sign_multipliers = np.where(signs == '-', -1.0, 1.0)
                        else:
                            sign_multipliers = np.ones(len(table))
                        # 3. Perform the exact mathematical conversion directly (Bypasses string formatting completely)
                        # RA: 1 hour = 15 degrees
                        ra_deg = (rah + (ram / 60.0) + (ras / 3600.0)) * 15.0
                        
                        # Dec: Calculate absolute magnitude, then apply the negative sign multiplier
                        dec_deg = sign_multipliers * (ded + (dem / 60.0) + (des / 3600.0))
                    
                        # 4. Safely update your table coordinates matrix
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
                    
                    except Exception as e:
                        print(f"Error during mathematical degree conversion: {e}")


                    #         # 1. Safely handle potential float/integer strings by stripping decimals (e.g., '12.0' -> '12')
                    #         rah = str(table['RAh'][i]).split('.')[0]
                    #         ram = str(table['RAm'][i]).split('.')[0]
                    #         ras = str(table['RAs'][i])  # Seconds can keep decimals
                            
                    #         # 2. Fix the bytes issue (e.g., b'-' -> '-') and strip decimals from degrees
                    #         sign_val = table['DE-'][i]
                    #         dec_sign = sign_val.decode('utf-8') if isinstance(sign_val, bytes) else str(sign_val)
                            
                    #         ded = str(table['DEd'][i]).split('.')[0]
                    #         dem = str(table['DEm'][i]).split('.')[0]
                    #         des = str(table['DEs'][i])  # Seconds can keep decimals
                            
                    #         # 3. Build clean coordinate strings
                    #         ra_str = f"{rah}:{ram}:{ras}"
                    #         dec_str = f"{dec_sign}{ded}:{dem}:{des}"
                    #         coordinate_strings.append(f"{ra_str} {dec_str}")
                    #     coords = SkyCoord(coordinate_strings, unit=(u.hourangle, u.deg), frame='icrs')
                    #     ra_deg = coords.ra.degree
                    #     dec_deg = coords.dec.degree
                    #     if 'RAJ2000' in table.colnames:
                    #         table.replace_column('RAJ2000', ra_deg)
                    #     else:
                    #         table.add_column(ra_deg, name='RAJ2000')
                            
                    #     if 'DEJ2000' in table.colnames:
                    #         table.replace_column('DEJ2000', dec_deg)
                    #     else:
                    #         table.add_column(dec_deg, name='DEJ2000')
                    #     table['RAJ2000'].unit = u.deg
                    #     table['DEJ2000'].unit = u.deg
                        
                    # except Exception as e:                            
                    #     print(e)

                    catalogs.append(table)
                    processed_names.append(name) # Save name for frequency lookups
                elif name == 'TGSS':
                    if 'RAJ2000' in table.colnames:
                        processed_names.append(name) # Save name for frequency lookups
                        catalogs.append(table)
                    # skip
                    else:
                        table.rename_columns(['RA', 'DEC'], ['RAJ2000', 'DEJ2000'])
                        processed_names.append(name) # Save name for frequency lookups
                        catalogs.append(table)


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
    # just dont write down mals for irght now
    # final_catalogs = catalogs + mals_catalogs
    # final_catalog_names = ['mals_all
    print()
    print(processed_names)
    print(all_paths)
    print()
    if len(catalogs) != len(all_paths):
        print('STOP SIZE MISMATCH')
    else:
        for (index, catalog), catalog_name in zip(enumerate(catalogs), processed_names):
            try:
                if catalog_name == 'SUMSS':
                    catalog.write(all_paths[index], format='fits', overwrite=True)
                elif catalog_name == 'NVSS':
                    catalog.write(all_paths[index], format='fits', overwrite=True)
                else:
                    catalog.write(all_paths[index], format='fits', overwrite=False)
                print(f'{catalog_name} write complete!')
            except Exception as e:
                print(e)
                continue

        print('catalog fits prepping done!')
                
    
if __name__ == '__main__':
    main()
