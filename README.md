# USSCataloging
NRAO Summer Project 2026
Currently includes a example script to run the overlay, only tested between MALS and FERMI-LAT14 surveys
- if you wish to change the ellipse overlay alter the final row of indexes fom 0 to another number
- if you wish to see all ellipses, make a for loop with the size of patches
- note that each file does get saved (overrides with the same file name)
Pre-Req Libraries:
numpy, astropy, astroquery, nway, xgboost, etc
alot of the methods belows suffer from incomplete steps or steps that contains significant bugs


## Path 1 - Recreating Our State:
### Catalog Download & Generation 
1. Download 8, 10, 12, 14, and 16yr fermi lats fits catalogs
2. Download racs-low, tgss, vlass, nvss, sumss, spice-racs from specified locations* (linked to table BLANK in paper
3. Download the WISE, PANSTARRS, GAIA Catalogs using the files in query_scripts** (** means very instensive
4. Use gaia_wise_pan_overlay.py to generate the overlapping catalogs of gaia, wise, and panstarr sources for a given fermi year between associated and unassociated sources
5. Standardize all the comparisions catalogs (or a new comparision catalog outside of the aformentioned) using  catalog_prep.py  
### Catalog Overlap & Reduction
7. generate the reduced associated catalogs and unassociated catalogs from years 8-16 using assoc_catalog_script_V3.py & unassoc_catalog_script.py (this could be combined into one file that is just terminal called instead!)
8. generate the specific sources that went from unassociated to associated using time_vary_fermi_error_script_V2.py
9. generate prior files using assoc_prior_generation.py from the 14yr associated catalog
### NWAY Run & Decision Tree Generation
10. run the nway matching of catalog years through nway_primer.py
11. generate a grouped table of the top three matches alongside the worst match per source for all the fermi ellipses across all of the fermi years using  top_matches.py 
12. create a subtable of the unassociated to associated sources from the nway processed catalog years 8-16yr (currently 8-14) in top_matches.py

13. create the gradiant boosted decision tree model using descision_tree.py (this will generate a best fit model and save it onto the cwd) (this will also do a self-test for accuracy and a test using the unassociated to associated sources through a 70-20-10 dataset split)
    - generates a 3d clustering of the predicted label to the true label
    - generates other analyitics about the measurment accuracy and various parameters + their meanings!
### Probability Measurement & Analytics 
14. Apply the unassociated sources in the 14yr catalog using fermi_plotting.py
    - will generate a histogram that overlays the unassociated source probabilites to the unassociated sources
    - generate a 3d clustering of the unassociated sources alongside a table of the driven features for the clustering based on umap
    - should be able to accept the individual fermi ellipse completley or the top three and worst match 
### Visualizations/Morgan Classification Scheme
14. Append morgan classification scheme to the top matches sources!
15. run the top matches with the probability overlay onto a visualization script i.e how morgan's is setup (astroquery) in fermi_plotting.py


## Path 2 - Calling a Specific Fermi Ellipse for Measuremnt:

## Path 3 - Running a New Fermi Ellipse or a Previously Studied Fermi Ellipse with New Matches Through Our Model:
