import glob
import os
import subprocess
from astropy.table import Table
from astropy.io import ascii
from tqdm import tqdm

input_pattern = "/lustre/aoc/observers/nm-16041/antonio/USSCataloging/catalogs/wise_temp/**/*.bz2"
output_fits = "/lustre/aoc/observers/nm-16041/antonio/USSCataloging/catalogs/combined_wise.fits"

files = sorted(glob.glob(input_pattern, recursive=True))
if not files:
    print("No compressed WISE files found.")
    exit()

print(f"Direct execution: Stacking {len(files)} files via Linux Stream engine.")

for f in tqdm(files, desc="Writing WISE FITS", unit="file"):
    try:
        # This completely completely bypasses the bug causing your crashes.
        process = subprocess.Popen(['bzcat', f], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, _ = process.communicate()
        
        # Read the raw text stream using the fast basic whitespace engine
        # We manually skip the IPAC meta lines ('\\' and '|') to protect formatting
        lines = [line for line in stdout.splitlines() if line.strip() and not line.startswith('\\') and not line.startswith('|')]
        
        if not lines:
            continue
            
        # Parse the clean text data array directly into an Astropy Table
        astropy_table = ascii.read(lines, format='basic', delimiter=' ')
        
        # Standard FITS header name lengths normalization loop
        for col_name in astropy_table.colnames:
            if len(col_name) > 68:
                short_name = col_name[:68]
                astropy_table.rename_column(col_name, short_name)

        # Append step
        if os.path.exists(output_fits):
            astropy_table.write(output_fits, format="fits", append=True)
        else:
            astropy_table.write(output_fits, format="fits")
            
    except Exception as e:
        print(f"\nError processing {os.path.basename(f)}: {e}")

print("Success! Process completely finished.")
