import glob
import os
import time
import pandas as pd
from astropy.table import Table, vstack
from tqdm import tqdm
import numpy as np

input_pattern = "/lustre/aoc/observers/nm-16041/USSCataloging/catalogs/*.csv.gz"
output_fits = "/lustre/aoc/observers/nm-16041/antonio/USSCataloging/catalogs/gaia.fits"

files = sorted(glob.glob(input_pattern))
if not files:
    print("No files found matching the pattern.")
    exit()

# Check existing progress directly inside the FITS file
existing_table = None
processed_files = set()
if os.path.exists(output_fits):
    try:
        existing_table = Table.read(output_fits, format="fits")
        if "origin_file" in existing_table.colnames:
            processed_files = set(str(name).strip() for name in existing_table["origin_file"])
        print(f"Resuming: Found {len(existing_table):,} existing sources from {len(processed_files)} files.")
    except Exception as e:
        print(f"Could not read existing FITS. Starting fresh. Error: {e}")
        existing_table = None

files_to_process = [f for f in files if os.path.basename(f) not in processed_files]
if not files_to_process:
    print("All files are already appended inside the FITS table!")
    exit()

print(f"Processing {len(files_to_process)} new files...")
new_tables = []
start_time = time.monotonic()

# Global fixed size mapping for final output file structure
STR_WIDTHS = {
    "designation": 35,
    "origin_file": 100,
    "DEFAULT": 50
}

# Process in larger chunks for significantly faster runtimes
SAVE_CHUNK_SIZE = 50

for idx, f in enumerate(tqdm(files_to_process, desc="Appending Gaia Sources", unit="file")):
    try:
        df = pd.read_csv(f, comment="#", low_memory=False)
        if df.empty:
            continue
        
        astropy_table = Table()
        for col_name in df.columns:
            series = df[col_name]
            
            # Keep ingestion basic and fast. Let types flex naturally.
            if series.dtype == 'object' or series.dtype.kind in {'S', 'U'}:
                astropy_table[col_name] = series.fillna("").astype(str).values
            else:
                astropy_table[col_name] = series.values
        
        # Track origin file tracking natively
        astropy_table["origin_file"] = np.array([os.path.basename(f)] * len(df), dtype=str)
        new_tables.append(astropy_table)
        
    except Exception as e:
        print(f"\nError reading {os.path.basename(f)}: {e}")

    # CHECKPOINT:
    if (idx + 1) % SAVE_CHUNK_SIZE == 0 and new_tables:
        tqdm.write("\n[Checkpoint] Consolidating batch and saving to FITS...")
        
        # VSTACK 1: Merge ONLY the new tables together first
        merged_new_batch = vstack(new_tables)
        
        # Flatten the text columns of the new batch to match the strict schema
        flat_new_batch = Table()
        for col_name in merged_new_batch.colnames:
            col_data = merged_new_batch[col_name]
            if col_data.dtype.kind in {'U', 'S', 'O', 'V'}:
                limit = STR_WIDTHS.get(col_name, STR_WIDTHS["DEFAULT"])
                flat_new_batch[col_name] = col_data.astype(f"<U{limit}")
            else:
                flat_new_batch[col_name] = col_data
        
        # VSTACK 2: Safely combine the existing table with the flattened new batch
        if existing_table is not None:
            existing_table = vstack([existing_table, flat_new_batch])
        else:
            existing_table = flat_new_batch
            
        # Write to disk cleanly
        existing_table.write(output_fits, format="fits", overwrite=True)
        
        # Reset the batch tracking array and clear RAM
        new_tables.clear()

# Final wrap-up save for any remaining files left in the queue
if new_tables:
    print("\nMerging final batch and updating FITS file...")
    merged_new_batch = vstack(new_tables)
    
    flat_new_batch = Table()
    for col_name in merged_new_batch.colnames:
        col_data = merged_new_batch[col_name]
        if col_data.dtype.kind in {'U', 'S', 'O', 'V'}:
            limit = STR_WIDTHS.get(col_name, STR_WIDTHS["DEFAULT"])
            flat_new_batch[col_name] = col_data.astype(f"<U{limit}")
        else:
            flat_new_batch[col_name] = col_data
            
    if existing_table is not None:
        final_table = vstack([existing_table, flat_new_batch])
    else:
        final_table = flat_new_batch
        
    final_table.write(output_fits, format="fits", overwrite=True)
    total_sources = len(final_table)
else:
    total_sources = len(existing_table) if existing_table is not None else 0

total_time = time.monotonic() - start_time
print(f"Successfully finished! Total sources now: {total_sources:,}")
print(f"Total time elapsed: {total_time:.2f} seconds.")
