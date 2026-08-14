# creates the associated catalog (one big table where you collate all the sources per fermi ellipse!)
from pprint import pprint
import glob
from copy import copy
import re
import os 

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
from astropy.table import vstack, hstack
# too hard to try and find all the quantities right now but need to fix later!
from astropy.table import QTable, Table, Column, MaskedColumn
from astropy.modeling import models, fitting
from astropy.coordinates import SkyCoord


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
import time

# frequencies should already be prepped
import glob
from astropy.table import QTable
import dask.dataframe as dd

import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import KDTree
# the associated sources are just directly appened whil the unassociated sources cannot be directly append nor can they look for the closest match
def main(fermi_type='associated', group_type='single'):
    current_working_dir = os.getcwd()
    catalog_home_dir = current_working_dir + '/catalogs/'
    catalog_reduced_dir = current_working_dir + '/reduced_catalogs/'
    catalog_mals_all = current_working_dir + '/catalogs/mals_all_bands.fits'

    # gaia and wise input paths
    #NEED TO RENAME GAIA
    allwise_file_path = catalog_home_dir + 'wise_temp/irsa.ipac.caltech.edu/data/download/wise-allwise/allwise_catalog.txt'
    gaia_fits_path = catalog_home_dir + 'gaia.fits'

    allwise_output_paths = []
    gaia_output_paths    = []
    # fermi lat surveys 
    #done matched files
    #just for associated files right now
    
    ellipse_grouped_tables = []
    if fermi_type == 'associated':
        # the name of the associated fermi file
        files = glob.glob(catalog_reduced_dir + group_type + '_14yr_associated_ellipses_all_srcs.fits')
        print(files)
        # change into being a list so it follows same for loop convention
        ellipse_grouped_table = QTable.read(files[0] , format ='fits')
        ellipse_grouped_tables.append(ellipse_grouped_table)

        # gaia|wise output path
        allwise_output_path = catalog_reduced_dir + group_type+ '_wise_matched_catalog_14yr_associated_ellipses_all_srcs.fits'
        gaia_output_path = catalog_reduced_dir + group_type + '_gaia_matched_catalog_14yr_associated_ellipses_all_srcs.fits'

        allwise_output_paths.append(allwise_output_path)
        gaia_output_paths.append(gaia_output_path)

    # reading the unassociated files
    else:
        fermi_lat_surveys = glob.glob(catalog_home_dir + 'gll**.fit')
        fermi_lat_surveys.sort(reverse=False)
        print(fermi_lat_surveys)
        catalog_years = [8, 10, 12, 14, 16]

        # TEMP
        # changing it temp fto do 14 year first 
        survery14_yr = fermi_lat_surveys.pop(-2)
        catalog_years_14 = catalog_years.pop(-2)
        # append to be front
        catalog_years.insert(0, catalog_years_14)
        fermi_lat_surveys.insert(0, survery14_yr)
        print(fermi_lat_surveys)
        print(catalog_years)

        
        
        # get tables fermi unassoc
        for index, fermi_survey in enumerate(fermi_lat_surveys):
            survey_hud = fits.open(fermi_survey)
            fermi_table = QTable.read(survey_hud , format ='fits')
            class1_mask = np.isin(fermi_table['CLASS1'], ['unk',''])
            fermi_unassoc_table = fermi_table[class1_mask]
            ellipse_grouped_tables.append(fermi_unassoc_table)
            # gaia|wise output path
            allwise_output_path = catalog_reduced_dir +group_type +f'_wise_matched_catalog_{catalog_years[index]}yr_unassociated_ellipses_all_srcs.fits'
            gaia_output_path = catalog_reduced_dir +group_type +f'_gaia_matched_catalog_{catalog_years[index]}yr_unassociated_ellipses_all_srcs.fits'

            allwise_output_paths.append(allwise_output_path)
            gaia_output_paths.append(gaia_output_path)

    print('output paths')
    print(gaia_output_paths)
    print(allwise_output_paths)
    # check if gaia wise table already exist!
    # should only be one file per iteration in the list if there is e.g [[a,b], c] then there is a issue
    allwise_existing_catalog_paths = []
    gaia_existing_catalog_paths    = []
    for path_allwise, path_gaia in zip(allwise_output_paths, gaia_output_paths):
        allwise_existing_catalog_path = glob.glob(path_allwise)
        gaia_existing_catalog_path    = glob.glob(path_gaia)

        # THIS IS A BUG!!! IT WILL SKIP IF OUT OF ORDER OR IF ONLY SOME HAVE PATHS BUT NOT OTHERS!
        if allwise_existing_catalog_path:
            allwise_existing_catalog_paths.append(allwise_existing_catalog_path)
        else:
            allwise_existing_catalog_paths.append('N/A')
            
        if gaia_existing_catalog_path:
            gaia_existing_catalog_paths.append(gaia_existing_catalog_path)
        else:
            gaia_existing_catalog_paths.append('N/A')


    print(allwise_existing_catalog_paths)
    print(gaia_existing_catalog_paths)
            

        # Exact 154-column sequence for the AllWISE raw file format
    allwise_columns = [
            # Basic Position and Identification Information
            "designation", "ra", "dec", "sigra", "sigdec", "sigradec", "glon", "glat", 
            "elon", "elat", "wx", "wy", "cntr", "source_id", "coadd_id", "src",
            
            # Primary Photometric Information
            "w1mpro", "w1sigmpro", "w1snr", "w1rchi2", 
            "w2mpro", "w2sigmpro", "w2snr", "w2rchi2", 
            "w3mpro", "w3sigmpro", "w3snr", "w3rchi2", 
            "w4mpro", "w4sigmpro", "w4snr", "w4rchi2", 
            "rchi2", "nb", "na", "w1sat", "w2sat", "w3sat", "w4sat", "satnum",
            
            # Motion Fit Parameters
            "ra_pm", "dec_pm", "sigra_pm", "sigdec_pm", "sigradec_pm", 
            "pmra", "sigpmra", "pmdec", "sigpmdec", 
            "w1rchi2_pm", "w2rchi2_pm", "w3rchi2_pm", "w4rchi2_pm", "rchi2_pm", "pmcode",
            
            # Measurement Quality and Source Reliability Information
            "cc_flags", "rel", "ext_flg", "var_flg", "ph_qual", "det_bit", "moon_lev",
            "w1nm", "w1m", "w2nm", "w2m", "w3nm", "w3m", "w4nm", "w4m",
            "w1cov", "w2cov", "w3cov", "w4cov",
            "w1cc_map", "w1cc_map_str", "w2cc_map", "w2cc_map_str",
            "w3cc_map", "w3cc_map_str", "w4cc_map", "w4cc_map_str",
            
            # Only present in the raw data stream / full catalog
            "use_src", "best_use_cntr", "ngrp",
            
            # Additional Photometric Information (Fluxes and backgrounds)
            "w1flux", "w1sigflux", "w1sky", "w1sigsk", "w1conf",
            "w2flux", "w2sigflux", "w2sky", "w2sigsk", "w2conf",
            "w3flux", "w3sigflux", "w3sky", "w3sigsk", "w3conf",
            "w4flux", "w4sigflux", "w4sky", "w4sigsk", "w4conf",
            
            # Multi-Aperture Photometry (Magnitudes 1 through 8)
            "w1mag", "w1sigm", "w1flg", "w1mcor",
            "w2mag", "w2sigm", "w2flg", "w2mcor",
            "w3mag", "w3sigm", "w3flg", "w3mcor",
            "w4mag", "w4sigm", "w4flg", "w4mcor",
            "w1mag_1", "w1sigm_1", "w1flg_1", "w2mag_1", "w2sigm_1", "w2flg_1",
            "w3mag_1", "w3sigm_1", "w3flg_1", "w4mag_1", "w4sigm_1", "w4flg_1",
            "w1mag_2", "w1sigm_2", "w1flg_2", "w2mag_2", "w2sigm_2", "w2flg_2",
            "w3mag_2", "w3sigm_2", "w3flg_2", "w4mag_2", "w4sigm_2", "w4flg_2",
            "w1mag_3", "w1sigm_3", "w1flg_3", "w2mag_3", "w2sigm_3", "w2flg_3",
            "w3mag_3", "w3sigm_3", "w3flg_3", "w4mag_3", "w4sigm_3", "w4flg_3",
            "w1mag_4", "w1sigm_4", "w1flg_4", "w2mag_4", "w2sigm_4", "w2flg_4",
            "w3mag_4", "w3sigm_4", "w3flg_4", "w4mag_4", "w4sigm_4", "w4flg_4",
            "w1mag_5", "w1sigm_5", "w1flg_5", "w2mag_5", "w2sigm_5", "w2flg_5",
            "w3mag_5", "w3sigm_5", "w3flg_5", "w4mag_5", "w4sigm_5", "w4flg_5",
            "w1mag_6", "w1sigm_6", "w1flg_6", "w2mag_6", "w2sigm_6", "w2flg_6",
            "w3mag_6", "w3sigm_6", "w3flg_6", "w4mag_6", "w4sigm_6", "w4flg_6",
            "w1mag_7", "w1sigm_7", "w1flg_7", "w2mag_7", "w2sigm_7", "w2flg_7",
            "w3mag_7", "w3sigm_7", "w3flg_7", "w4mag_7", "w4sigm_7", "w4flg_7",
            "w1mag_8", "w1sigm_8", "w1flg_8", "w2mag_8", "w2sigm_8", "w2flg_8",
            "w3mag_8", "w3sigm_8", "w3flg_8", "w4mag_8", "w4sigm_8", "w4flg_8",
            
            # Variability Diagnostics
            "w1magp", "w1sigp1", "w1sigp2", "w1k", "w1ndf", "w1mlq", "w1mjdmin", "w1mjdmax", "w1mjdmean",
            "w2magp", "w2sigp1", "w2sigp2", "w2k", "w2ndf", "w2mlq", "w2mjdmin", "w2mjdmax", "w2mjdmean",
            "w3magp", "w3sigp1", "w3sigp2", "w3k", "w3ndf", "w3mlq", "w3mjdmin", "w3mjdmax", "w3mjdmean",
            "w4magp", "w4sigp1", "w4sigp2", "w4k", "w4ndf", "w4mlq", "w4mjdmin", "w4mjdmax", "w4mjdmean",
            "rho12", "rho23", "rho34", "q12", "q23", "q34",
            
            # 2MASS PSC Association Information
            "xscprox", "w1rsemi", "w1ba", "w1pa", "w1gmag", "w1gerr", "w1gflg",
            "w2rsemi", "w2ba", "w2pa", "w2gmag", "w2gerr", "w2gflg",
            "w3rsemi", "w3ba", "w3pa", "w3gmag", "w3gerr", "w3gflg",
            "w4rsemi", "w4ba", "w4pa", "w4gmag", "w4gerr", "w4gflg",
            "tmass_key", "r_2mass", "pa_2mass", "n_2mass", 
            "j_m_2mass", "j_msig_2mass", "h_m_2mass", "h_msig_2mass", "k_m_2mass", "k_msig_2mass",
            
            # Spatial Indexing Information
            "x", "y", "z", "spt_ind", "htm20"
    ]
        #add the wise and gaia sources here!
    
    # The only 10 fields classified under string/character notation in the AllWISE documentation
    allwise_string_cols = [
        "designation",   # %20s - Sexagesimal source name
        "source_id",     # %28s - Unique source ID string
        "coadd_id",      # %20s - Atlas Tile identifier string
        "satnum",        # %4s  - Minimum saturation sample flag per band
        "pmcode",        # %5s  - Blend and distance encoded string
        "cc_flags",      # %4s  - Contamination and confusion flag string
        "rel",           # %1c  - Small-separation same-tile detection flag
        "ph_qual",       # %4s  - Photometric quality flag summary string
        "moon_lev",      # %4s  - Scattered moonlight contamination flag string
        "w1cc_map_str"   # %9s  - W1 contamination and confusion layout string
    ]

    # =============================================================================================================================
    # --- PRE-CALCULATE NUMPY ARRAYS FOR MAXIMUM VECTORIZATION SPEED ---
    # We convert table columns to raw numpy arrays so matching checks take microseconds.
    
    # Structure data to store matches 
    # We map matching lines directly to the specific Fermi row index they belong to 
    for table_index, ellipse_grouped_table in enumerate(ellipse_grouped_tables):
        
        fermi_ra = np.array(ellipse_grouped_table['RAJ2000'])
        fermi_dec = np.array(ellipse_grouped_table['DEJ2000'])
        fermi_radius = np.array(ellipse_grouped_table['Conf_95_SemiMajor'])
        
        # Bounded search vectors 
        ra_min = fermi_ra - fermi_radius 
        ra_max = fermi_ra + fermi_radius 
        dec_min = fermi_dec - fermi_radius 
        dec_max = fermi_dec + fermi_radius 
        
        GLOBAL_RA_MIN = float(np.min(ra_min))
        GLOBAL_RA_MAX = float(np.max(ra_max))
        GLOBAL_DEC_MIN = float(np.min(dec_min))
        GLOBAL_DEC_MAX = float(np.max(dec_max))
        
        if fermi_type=='associated':
            print(f'====Analysis on Associated-14-Yr-Fermi Catalog====')
        else:
            print(f'====Analysis on Unassociated-{catalog_years[table_index]}-Yr-Fermi Catalog====')
        print('START ALL WISE MATCHING')
        
        ellipse_matches = {i: [] for i in range(len(ellipse_grouped_table))} 
        file_size_bytes = os.path.getsize(allwise_file_path) 
        start_time = time.time() 
        lines_scanned = 0 
        
        # simple check to see if the full catalog match exists 
        if allwise_existing_catalog_paths[table_index] == 'N/A':
            print(f"Initializing single-pass binary stream... Scanning {file_size_bytes / (1024**3):.2f} GB of data.") 
        
            local_ra_min = ra_min
            local_ra_max = ra_max
            local_dec_min = dec_min
            local_dec_max = dec_max
            # Open file in binary stream mode to optimize I/O reading speeds 
            with open(allwise_file_path, 'rb') as f: 
                for line in f:
                    lines_scanned += 1 
                    #DEBUG - break to get early test for source matching 
                    # if lines_scanned % 5000001 == 0: 
                    #     break

                    # Periodic runtime diagnostic summary matrix 
                    if lines_scanned % 10000000 == 0: 
                    # if lines_scanned % 5000000 == 0: 
                        elapsed = time.time() - start_time 
                        current_offset = f.tell() 
                        scanned_gb = current_offset / (1024**3) 
                        total_gb = file_size_bytes / (1024**3) 
                        speed = scanned_gb / elapsed if elapsed > 0 else 0 
                        print(f"Scanned: {lines_scanned:,} lines | Progress: {scanned_gb:.1f}/{total_gb:.1f} GB | Stream-Speed: {speed:.2f} GB/s") 
                        matched_count = sum(1 for matches in ellipse_matches.values() if len(matches) > 0)
                        print(f"Found AllWISE match for {matched_count} Fermi Ellipses")
                        # the -1 is for the empty list!
                    
                    # Split out only the first 3 elements (Designation, RA, Dec). Drops 151 columns instantly! 
                    parts = line.split(b'|', 3) 
                    if len(parts) >= 3: 
                        try: 
                            ra = float(parts[1])
                            # QUICK CHECK 1: If RA is completely out of range of all ellipses, skip immediately!
                            if ra < GLOBAL_RA_MIN or ra > GLOBAL_RA_MAX:
                                continue
                                
                            dec = float(parts[2])
                            # QUICK CHECK 2: If Dec is completely out of range of all ellipses, skip immediately!
                            if dec < GLOBAL_DEC_MIN or dec > GLOBAL_DEC_MAX:
                                continue
                            
                            # VECTORIZED MATCHING: Run ONLY if the coordinates pass the global filters
                            matched_indices = np.where(
                                (ra >= local_ra_min) & (ra <= local_ra_max) &
                                (dec >= local_dec_min) & (dec <= local_dec_max)
                            )[0]
                            
                            # If it falls into any of our target ellipses, save the full row
                            if len(matched_indices) > 0:
                                decoded_line = line.decode('utf-8').strip().split('|')
                                #DEBUG
                                # print(decoded_line)
                                for idx in matched_indices:
                                    ellipse_matches[idx].append(decoded_line)
        
    
                                    
                        except ValueError: 
                            pass # Safely bypass headers or malformed text blocks 
                                
            total_time = time.time() - start_time 
            print(f"\nSingle-Pass AllWISE Scan Complete in {total_time:.2f} seconds. Parsed {lines_scanned:,} rows.") 
            
            # --- CONVERT MATCHES INTO ASTROPY ASSOCIATIONS --- 
            # We will create an array matching the size of ellipse_grouped_table to hstack cleanly 
            all_wise_matched_rows = [] 
            
            print("Processing target separations and sorting best cross-matches...") 
            for i in range(len(ellipse_grouped_table)):
                raw_rows = ellipse_matches[i]
                
                if len(raw_rows) == 0:
                    # 1. NO MATCH FOUND: Build a clean dictionary where strings get "" and numeric fields get np.nan
                    empty_row = {}
                    for col in allwise_columns:
                        if col in allwise_string_cols:
                            empty_row[col] = ""  # Safe blank string placeholder
                        else:
                            empty_row[col] = np.nan  # Numeric columns natively support floating-point NaN
                    empty_row['wise_ang_sep'] = np.nan
                    all_wise_matched_rows.append(empty_row)
                    
                else:
                    # 2. MATCHES FOUND: Pre-convert string chunks into explicit Python primitives
                    cleaned_candidates = []
                    for raw_line_split in raw_rows:
                        cleaned_row_dict = {}
                        for col_idx, col_name in enumerate(allwise_columns):
                            val_str = raw_line_split[col_idx]
                            
                            # Check for AllWISE explicit 'null' strings or missing markers
                            if val_str.lower() in ['null', '----', '', 'NaN']:
                                if col_name in allwise_string_cols:
                                    cleaned_row_dict[col_name] = ""
                                else:
                                    cleaned_row_dict[col_name] = np.nan
                            else:
                                # Apply structural type casting
                                if col_name in allwise_string_cols:
                                    cleaned_row_dict[col_name] = str(val_str)
                                else:
                                    try:
                                        cleaned_row_dict[col_name] = float(val_str)
                                    except ValueError:
                                        cleaned_row_dict[col_name] = np.nan  # Fallback catch for unexpected characters
                        
                        cleaned_candidates.append(cleaned_row_dict)
                    # Build a temporary Astropy table out of pre-typed dictionaries
                    tmp_table = Table(cleaned_candidates)
                    #DEBUG
                    # Calculate precise angular separation values natively as floats
                    catalog_coords = SkyCoord(ra=tmp_table['ra'] * u.deg, dec=tmp_table['dec'] * u.deg)
                    target_coord = SkyCoord(ra=ellipse_grouped_table['RAJ2000'][i], dec=ellipse_grouped_table['DEJ2000'][i])
                                        
                    separations = target_coord.separation(catalog_coords)
                    tmp_table['wise_ang_sep'] = separations.to(u.arcsec).value
                    
                    # Sort values cleanly by angular distance
                    tmp_table.sort('wise_ang_sep')
                    tmp_table['Source_Name'] = [ellipse_grouped_table[i]['Source_Name']] * len(tmp_table)

                    if group_type=='single':
                        # Turn the closest matching row back into a clean typed record row dictionary
                        closest_match_dict = dict(tmp_table[0])
                        all_wise_matched_rows.append(closest_match_dict)
                    elif group_type == 'grouped':
                        # Loop through every row in the sorted tmp_table and append each individually
                        for row in tmp_table:
                            row_dict = {col: row[col] for col in tmp_table.colnames}
                            all_wise_matched_rows.append(row_dict)

            # Convert all synchronized flat rows into a properly structured Astropy Table
            # CRITICAL: Initialize with masked=True to handle any NaNs or missing entries properly
            allwise_final_table = Table(all_wise_matched_rows, masked=True)
            allwise_final_table['catalog_names'] = ['WISE'] * len(allwise_final_table)

            # Ensure data types are printed cleanly for verification
            #DEBUG
            print("\n--- SANITY CHECK: VERIFYING MASTER COLUMN DATA TYPES ---")
            for col_name in allwise_final_table.colnames:
                if allwise_final_table[col_name].dtype == 'O':
                    print(f"Column: {col_name:<15} | Type Kind: {allwise_final_table[col_name].dtype.kind}")
            
            # --- writing the all_wise table so if it fails dont have to rewrite it --- #
            try:
                # write it here since I dont want to double write the wise in the nway matching
                # automate in the future to write at specific path from the file it is looking at if it doesnt exist!
                print(f'catalog_size of true matches: {len(allwise_final_table[~allwise_final_table['designation'].mask])}')
                allwise_final_table.write(allwise_output_paths[table_index], format='fits', overwrite=False)
            except Exception as e:
                print(e)
                pass
                           
            # --- RESOLVE COLUMN CONFLICTS AND PREFIX MAP ---
            for col in allwise_final_table.colnames:
                if col not in ['wise_ang_sep', 'catalog_names']:
                    allwise_final_table.rename_column(col, f'wise_{col}')
        
        else:
            path = allwise_existing_catalog_paths.pop(table_index)
            # pass

            try:
                allwise_final_table = Table.read(path[0], format='fits')
                # --- RESOLVE COLUMN CONFLICTS --- 
                # Add a unique prefix to every AllWISE column name so they do not overwrite Fermi or Gaia columns 
                for col in allwise_final_table.colnames: 
                    if col not in ['wise_ang_sep', 'catalog_names']:
                        allwise_final_table.rename_column(col, f'wise_{col}') 
                
            except Exception as e:
                print(e)
                pass
        print('ALL WISE DONE')

# '''=========================================================GAIA======================================================================='''
        # there is some bug but this auto gen runs alot faster than the other way so youre gonna keep it like this!
        # if gaia_existing_catalog_paths[table_index] == 'N/A':
        #     # TEMP FOR NOW
        #     # continue
            
        #     print('START GAIA MATCHING')
        #     start_time = time.monotonic()
        #     # 1. Open Gaia using memory mapping
        #     with fits.open(gaia_fits_path, memmap=True) as hdul:
        #         gaia_data = hdul[1].data
        #         gaia_column_names = gaia_data.names
                
        #         print("Streaming Gaia coordinate blocks...")
        #         gaia_ra = np.array(gaia_data.field('ra'))
        #         gaia_dec = np.array(gaia_data.field('dec'))
                
        #         # 2. VECTORIZED SPHERICAL TO 3D CARTESIAN CONVERSION
        #         # We convert ALL 33+ million coordinates to Cartesian coordinates at once natively in NumPy
        #         print("Building optimized 3D Spatial K-D Tree Index...")
        #         rarad = np.radians(gaia_ra)
        #         decrad = np.radians(gaia_dec)
        #         gx = np.cos(decrad) * np.cos(rarad)
        #         gy = np.cos(decrad) * np.sin(rarad)
        #         gz = np.sin(decrad)
                
        #         # Construct the spatial tree structure
        #         gaia_tree = KDTree(np.column_stack((gx, gy, gz)))
        #         print(f"Tree built in {time.monotonic() - start_time:.1f} seconds.")
                
        #         # 3. VECTORIZED FERMI CARTESIAN CONVERSION
        #         print(f"Projecting {len(ellipse_grouped_table)} Fermi targets...")
        #         frad_ra = np.radians(fermi_ra)
        #         frad_dec = np.radians(fermi_dec)
        #         fx = np.cos(frad_dec) * np.cos(frad_ra)
        #         fy = np.cos(frad_dec) * np.sin(frad_ra)
        #         fz = np.sin(frad_dec)
        #         fermi_coords_3d = np.column_stack((fx, fy, fz))
                
        #         # Convert the angular radius limits into a direct linear 3D chord distance
        #         # chord_dist = 2 * sin(angular_radius / 2)
        #         chord_radii = 2 * np.sin(np.radians(fermi_radius) / 2)
                
        #         # 4. EXECUTE HIGH-SPEED SPATIAL TREE LOOKUPS
        #         print("Querying K-D Tree index for matches...")
        #         # query_ball_point searches all 33 million sources instantly via the tree index
        #         matched_indices_list = gaia_tree.query_ball_point(fermi_coords_3d, chord_radii)
                
        #         # Create an empty template table structure
        #         gaia_template = Table(gaia_data[0:1])
        #         gaia_template.remove_row(0)
                
        #         final_gaia_row_list = []
                
        #         print("Parsing query tree outputs and assigning separations...")
        #         for i in range(len(ellipse_grouped_table)):
        #             matched_indices = matched_indices_list[i]
                    
        #             if len(matched_indices) == 0:
        #                 # No matches found: Build empty row profile dictionary
        #                 empty_row = {col: (np.nan if gaia_template[col].dtype.kind in ['f', 'i'] else "") for col in gaia_column_names}
        #                 empty_row['gaia_ang_sep'] = np.nan
        #                 final_gaia_row_list.append(empty_row)
        #             else:
        #                 # Extract only the targeted match indexes from disk
        #                 matched_records = gaia_data[matched_indices]
        #                 tmp_table = Table(matched_records)
                        
        #                 # High-speed localized separation check
        #                 # Because we are processing a tiny subset of rows here, SkyCoord is fast
        #                 catalog_coords = SkyCoord(ra=tmp_table['ra']*u.deg, dec=tmp_table['dec']*u.deg)
        #                 target_coord = SkyCoord(ra=fermi_ra[i]*u.deg, dec=fermi_dec[i]*u.deg)
        #                 separations = target_coord.separation(catalog_coords)
        #                 tmp_table['gaia_ang_sep'] = separations.to(u.arcsec).value
                        
        #                 # Sort and filter for the closest target coordinate match
        #                 tmp_table.sort('gaia_ang_sep')
        #                 tmp_table['Source_Name'] = [ellipse_grouped_table[i]['Source_Name']] * len(tmp_table)
                        
        #                 if group_type == 'single':
        #                     closest_row_dict = {name: tmp_table[name][0] for name in tmp_table.colnames}
        #                     final_gaia_row_list.append(closest_row_dict)
        #                 elif group_type == 'grouped':
        #                     final_gaia_row_list.append(dict(tmp_table))
                            
        #         total_time = time.monotonic() - start_time
        #         print(f"Finished successfully! Total matching time elapsed: {total_time:.2f} seconds.")
        #         print(final_gaia_row_list)

        # # if gaia_existing_catalog_paths[table_index] == 'N/A':
        # #     print('START GAIA MATCHING') 
        # #     # Open the 1.95 TB Gaia file instantly via Memory Mapping
        # #     with fits.open(gaia_fits_path, memmap=True) as hdul:
        # #         gaia_data = hdul[1].data  # Index 1 contains the main binary table data
                
        # #         # Extract only the coordinate vectors from disk
        # #         print("Streaming Gaia coordinate blocks into virtual memory...")
        # #         gaia_ra = gaia_data.field('ra')
        # #         gaia_dec = gaia_data.field('dec')
                
        # #         # Get all column names
        # #         gaia_column_names = gaia_data.names
        # #         # print(gaia_column_names)
                
        # #         # --- FIX STEP 1: CREATE A TYPE-PERFECT TEMPLATE TABLE ---
        # #         # We read a single dummy row to inherit the perfect binary database data types
        # #         print("Creating synchronized schema template...")
        # #         gaia_template = Table(gaia_data[0:1])
        # #         gaia_template.remove_row(0)  # Empty it out, keeping only the perfect type headers
                
        # #         # Manually add the angular separation column to the type definition schema template
        # #         gaia_template['gaia_ang_sep'] = np.array([], dtype=float)
                
        # #         # This master list will collect individual typed Row objects
        # #         final_gaia_row_list = []
        # #         total_gaia_sources = len(gaia_data)

        # #         start_time = time.monotonic()
        # #         last_report = start_time
        # #         print(f"Looping through {len(ellipse_grouped_table)} Fermi entries against Gaia spatial arrays...")
                
        # #         for i in range(len(ellipse_grouped_table)):
        # #             # Time-based progress reporting instead of every-100 index checkpoints
        # #             now = time.monotonic()
        # #             if now - last_report >= 5:
        # #                 elapsed = now - start_time
        # #                 rate = (i + 1) / elapsed
        # #                 print(f"Cross-Matched: {i + 1}/{total_gaia_sources} sources "
        # #                           f"({rate:.1f} sources/s, {elapsed:.0f}s elapsed)...")
        # #                 last_report = now

                    
        # #             # Spatial bounding box query
        # #             matched_indices = np.where(
        # #                 (gaia_ra >= fermi_ra[i] - fermi_radius[i]) & (gaia_ra <= fermi_ra[i] + fermi_radius[i]) &
        # #                 (gaia_dec >= fermi_dec[i] - fermi_radius[i]) & (gaia_dec <= fermi_dec[i] + fermi_radius[i])
        # #             )[0]

        # #             if len(matched_indices) == 0:
        # #                 # --- FIX: Build a clean dictionary record instead of an invalid Row object ---
        # #                 empty_row = {}
                        
        # #                 # Fill numeric columns with np.nan and text/flag columns with empty strings
        # #                 for col in gaia_column_names:
        # #                     if gaia_template[col].dtype.kind in ['f', 'i']:
        # #                         empty_row[col] = np.nan
        # #                     else:
        # #                         empty_row[col] = ""
                                
        # #                 # Ensure the separation metric is included as a numeric nan
        # #                 empty_row['gaia_ang_sep'] = np.nan
        # #                 final_gaia_row_list.append(empty_row)
                        
        # #             else:
        # #                 # MATCHES FOUND: Extract only the matched records from the 1.95 TB database
        # #                 matched_records = gaia_data[matched_indices]
        # #                 tmp_table = Table(matched_records)
                        
        # #                 # Calculate precise physical angular separations
        # #                 catalog_coords = SkyCoord(ra=tmp_table['ra']*u.deg, dec=tmp_table['dec']*u.deg)
        # #                 target_coord = SkyCoord(ra=fermi_ra[i]*u.deg, dec=fermi_dec[i]*u.deg)
                        
        # #                 separations = target_coord.separation(catalog_coords)
        # #                 tmp_table['gaia_ang_sep'] = separations.to(u.arcsec).value
                        
        # #                 # Sort the local table by separation and choose index 0 (the closest match)
        # #                 tmp_table.sort('gaia_ang_sep')
        # #                 tmp_table['Source_Name'] = [ellipse_grouped_table[i]['Source_Name']] * len(tmp_table)

        # #                 if group_type=='single':
        # #                     # Turn the closest matching row back into a clean typed record row dictionary
        # #                     closest_row_dict = {name: tmp_table[name][0] for name in tmp_table.colnames}
        # #                     final_gaia_row_list.append(closest_row_dict)
        # #                 elif group_type=='grouped':
        # #                     final_gaia_row_list.append(dict(tmp_table))
                            
          
        #     # --- FIX STEP 3: INITIALIZE MASTER TABLE FROM TEMPLATE SCHEMA ---
        #     # Passing rows directly to a structured template forces all entries to align types
        #     gaia_final_table = Table(rows=final_gaia_row_list, names=gaia_template.colnames)
        #     # Bug Fix: Set catalog_names length safely to match gaia_final_table instead of allwise_final_table
        #     gaia_final_table['catalog_names'] = ['GAIA'] * len(gaia_final_table)
            
        #     print("\n--- SANITY CHECK: GAIA COLUMN DATA TYPES PRESERVED ---")
        #     for col_name in gaia_final_table.colnames:
        #         print(f"Gaia Column: {col_name:<15} | Type Kind: {gaia_final_table[col_name].dtype.kind}")
                
        #         if gaia_final_table[col_name].dtype == 'O':
        #             print(f"Gaia Column: {col_name:<15} | Type Kind: {gaia_final_table[col_name].dtype.kind}")
                    

        #     try:
        #         gaia_final_table.write(gaia_output_paths[table_index], format='fits', overwrite=False)
        #     except Exception as e:
        #         print(e)
        #         pass
        
        #     # --- RESOLVE COLUMN CONFLICTS ---
        #     # Add a unique prefix to every Gaia column name so they do not overwrite Fermi columns
        #     for col in gaia_final_table.colnames:
        #         if col not in ['gaia_ang_sep', 'catalog_names']:
        #             gaia_final_table.rename_column(col, f'gaia_{col}')
                
        # else:
        #     # popping will reduce it for the next iteration
        #     path = gaia_existing_catalog_paths.pop(table_index)

        #     try:
        #         gaia_final_table = Table.read(path, format='fits')
                         
        #         # --- RESOLVE COLUMN CONFLICTS ---
        #         # Add a unique prefix to every Gaia column name so they do not overwrite Fermi columns
        #         for col in gaia_final_table.colnames:
        #             if col not in ['gaia_ang_sep', 'catalog_names']:
        #                 gaia_final_table.rename_column(col, f'gaia_{col}')
                    
        #     # write it here since I dont want to double write the wise in the nway matching
        #     # automate in the future to write at specific path from the file it is looking at if it doesnt exist!
        #     except Exception as e:
        #         print(e)
        #         pass
        #     print('GAIA DONE')
    # '''==================================================================FINAL=MATCHING======================================================'''
        
        # Use hstack safely since both structures are now perfectly aligned row-for-row
        if group_type == 'single':
            # ellipse_grouped_table = hstack([ellipse_grouped_table, allwise_final_table, gaia_final_table])
            ellipse_grouped_table = hstack([ellipse_grouped_table, allwise_final_table])

            # Export the final fully finished masked matrix to your FITS path
            final_output_path = catalog_reduced_dir + '14' + 'yr_associated_ellipses_all_srcs_gaia_wise.fits'
            ellipse_grouped_table.write(final_output_path, format='fits', overwrite=True)
        elif group_type=='grouped':
            # continue
            # going to stack manually! as of 8/2/2026 change in future
            # ellipse_grouped_table = vstack([ellipse_grouped_table, allwise_final_table, gaia_final_table])
            ellipse_grouped_table = vstack([ellipse_grouped_table, allwise_final_table])
            final_output_path = catalog_reduced_dir + '14' + 'yr_associated_ellipses_all_srcs_gaia_wise_background.fits'
            ellipse_grouped_table.write(final_output_path, format='fits', overwrite=True)

        else:
            continue

        if fermi_type =='unassociated':
            print(f"Successfully finalized gaia and wise catalogs for {catalog_years[table_index]}")
        else:
            print(f"Successfully finalized gaia and wise catalogs for 14yr")

        # print(f"14yr-fermi catalogs complete")

    print('all done!')
if __name__ == '__main__':
    main(fermi_type = 'associated', group_type='single')
