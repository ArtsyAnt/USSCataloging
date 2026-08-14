# Goal: get the top nway matches within our dataset and make a catalog of them
import glob 
from astropy.io import fits
from astropy.table import vstack
from astropy.table import QTable, Table, Column, MaskedColumn
import os
import numpy as np
import time

# take input of whether we want just distance matching sources, or mag-prior varient
# currently only for the 14 year call
cols_to_remove = [
'WISE_w1mag', 'WISE_w1sigm', 'WISE_w1flg', 'WISE_w1mcor',
'WISE_w2mag', 'WISE_w2sigm', 'WISE_w2flg', 'WISE_w2mcor',
'WISE_w3mag', 'WISE_w3sigm', 'WISE_w3flg', 'WISE_w3mcor',
'WISE_w4mag', 'WISE_w4sigm', 'WISE_w4flg', 'WISE_w4mcor',

# --------------------------------------------------------------------------
# IX. ALLWISE MULTI-APERTURE STEPS ARRAY (APERTURES 1 THROUGH 8)
# --------------------------------------------------------------------------
'WISE_w1mag_1', 'WISE_w1sigm_1', 'WISE_w1flg_1', 'WISE_w2mag_1', 'WISE_w2sigm_1', 'WISE_w2flg_1', 'WISE_w3mag_1', 'WISE_w3sigm_1', 'WISE_w3flg_1', 'WISE_w4mag_1', 'WISE_w4sigm_1', 'WISE_w4flg_1',
'WISE_w1mag_2', 'WISE_w1sigm_2', 'WISE_w1flg_2', 'WISE_w2mag_2', 'WISE_w2sigm_2', 'WISE_w2flg_2', 'WISE_w3mag_2', 'WISE_w3sigm_2', 'WISE_w3flg_2', 'WISE_w4mag_2', 'WISE_w4sigm_2', 'WISE_w4flg_2',
'WISE_w1mag_3', 'WISE_w1sigm_3', 'WISE_w1flg_3', 'WISE_w2mag_3', 'WISE_w2sigm_3', 'WISE_w2flg_3', 'WISE_w3mag_3', 'WISE_w3sigm_3', 'WISE_w3flg_3', 'WISE_w4mag_3', 'WISE_w4sigm_3', 'WISE_w4flg_3',
'WISE_w1mag_4', 'WISE_w1sigm_4', 'WISE_w1flg_4', 'WISE_w2mag_4', 'WISE_w2sigm_4', 'WISE_w2flg_4', 'WISE_w3mag_4', 'WISE_w3sigm_4', 'WISE_w3flg_4', 'WISE_w4mag_4', 'WISE_w4sigm_4', 'WISE_w4flg_4',
'WISE_w1mag_5', 'WISE_w1sigm_5', 'WISE_w1flg_5', 'WISE_w2mag_5', 'WISE_w2sigm_5', 'WISE_w2flg_5', 'WISE_w3mag_5', 'WISE_w3sigm_5', 'WISE_w3flg_5', 'WISE_w4mag_5', 'WISE_w4sigm_5', 'WISE_w4flg_5',
'WISE_w1mag_6', 'WISE_w1sigm_6', 'WISE_w1flg_6', 'WISE_w2mag_6', 'WISE_w2sigm_6', 'WISE_w2flg_6', 'WISE_w3mag_6', 'WISE_w3sigm_6', 'WISE_w3flg_6', 'WISE_w4mag_6', 'WISE_w4sigm_6', 'WISE_w4flg_6',
'WISE_w1mag_7', 'WISE_w1sigm_7', 'WISE_w1flg_7', 'WISE_w2mag_7', 'WISE_w2sigm_7', 'WISE_w2flg_7', 'WISE_w3mag_7', 'WISE_w3sigm_7', 'WISE_w3flg_7', 'WISE_w4mag_7', 'WISE_w4sigm_7', 'WISE_w4flg_7',
'WISE_w1mag_8', 'WISE_w1sigm_8', 'WISE_w1flg_8', 'WISE_w2mag_8', 'WISE_w2sigm_8', 'WISE_w2flg_8', 'WISE_w3mag_8', 'WISE_w3sigm_8', 'WISE_w3flg_8', 'WISE_w4mag_8', 'WISE_w4sigm_8', 'WISE_w4flg_8',
]

def main(match_type='mag-prior', top_x_matches = None, skip_existing_cats=False):
    '''CATALOG CALLS'''
    current_working_dir = os.getcwd()
    catalog_nway_matches = current_working_dir + '/reduced_catalogs/'
    # glob search all the nway matches
    if match_type =='mag-prior':
        match_name = 'mag_prior_all_nway_matches/'
    elif match_type == 'distance':
        match_name = 'distance_all_nway_matches/'
    
    # cat_years = ['8', '10', '12', '14']
    cat_years = ['14']
    for cat_year in cat_years:
        if skip_existing_cats==True:
            stacked_catalog = glob.glob(catalog_nway_matches + match_name + cat_year +  '/all_top_tables_stacked_probabilites.fits')
            if len(stacked_catalog) >0:
                print(f'skipping {cat_year}-Yr Catalog')
                continue

        print(f'====Top Probability Read on {cat_year}-Yr-Fermi Catalog====')
        # base dir
            
        all_paths = glob.glob(catalog_nway_matches + match_name + cat_year + '/*.fits')
        # initiate all the files (takes a long time)
        # take the top p_i and p_any of each catalog and stack them into a new catalog for the fermi_ellipse
        all_top_tables = []
        start_time = time.monotonic()
        total_table_idx = len(all_paths)
        last_report = start_time
        for path_idx,  path in enumerate(all_paths):
            table = Table.read(path, format = 'fits')
            # absolute is not a good metric  
            # mask out only the top matches 
            # TODO: ADD SO IT IS THE TOP THREE right now gets the top and worse match 
            max_p_any_prob = max(table['prob_has_match'])
            table = table[table['prob_has_match'] == max_p_any_prob]
    
            # do arg max of the p_i
            # make the argmax the top and the lowest one
            max_props_idx = np.argmax(table['prob_this_match'])
            min_props_idx = np.argmin(table['prob_this_match'])

            # create a new table from that sublist 
            table = Table(table[[max_props_idx, min_props_idx]])
            all_top_tables.append(table)

            # Time-based progress reporting instead of every-100 index checkpoints
            now = time.monotonic()
            if now - last_report >= 5:
                elapsed = now - start_time
                rate = (path_idx + 1) / elapsed
                print(f"Cross-Matched: {path_idx + 1}/{total_table_idx} sources "
                          f"({rate:.1f} sources/s, {elapsed:.0f}s elapsed)...")
                last_report = now
    
            
        print(f"\nmax & min table sorting complete in {time.monotonic() - start_time:.0f}s. ")
        # vstack tables & write this catalog 
        all_top_tables_stacked = vstack(all_top_tables)
        print(len(all_top_tables_stacked.colnames))
        print()

        all_top_tables_stacked.remove_columns(cols_to_remove)
        print(len(all_top_tables_stacked.colnames))

        all_top_tables_stacked.write(catalog_nway_matches + match_name + cat_year +'/all_top_tables_stacked_probabilites.fits', format='fits', overwrite=True)
        print('Writing complete')

        # all_top_tables_stacked_across_years.append(all_top_tables_stacked)
        
    print('all writing complete')
    print('creating subtable of the unassociated to associated sources')

    
if __name__ == '__main__':
    main(skip_existing_cats=True)