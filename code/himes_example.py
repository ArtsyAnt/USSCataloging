# example of Fermi Unassociated Sources in the MeerKAT Absorption Line Survey catalog reduction using classes
# this should use the 3rd data release which is v31 not v35
from multiwavecataloging import Catalog
from multiwavecataloging import CatalogOverlayer
from multiwavecataloging import fermi_plot
from multiwavecataloging import spectral_window_merging
from multiwavecataloging import global_index

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.io import fits
from astropy.table import vstack
from astropy.table import Table, Column, MaskedColumn
# from astropy.table import vstack

from astropy import wcs
from regions import EllipseSkyRegion

# Access astronomical databases
from astroquery.vizier import Vizier

# For plots
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as ellipse

# Data handling
import numpy as np


def main():    
    # catalog call
    vizier = Vizier(columns=[])
    vizier.ROW_LIMIT = -1# -1 means unlimited
    
    '''CATALOG CALLS'''
    mals_catalog = vizier.get_catalogs('J/ApJS/270/33')[0]
    # changed fermi data call - 6/12
    Fermi_hud = fits.open('/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/gll_psc_v31.fit')
    Fermi_catalog = Table(Fermi_hud[1].data)
    
    # make sure fermi is only calling unassociated sources
    all_associations_types = list(set(Fermi_catalog['CLASS1']))
    unknown_associated_sources = [class_type for class_type in all_associations_types if class_type.islower()]

    class_type_mask = np.isin(Fermi_catalog['CLASS1'], unknown_associated_sources)
    Fermi_catalog = Fermi_catalog[class_type_mask]

    # reordering (want to put this into)
    # weird its double counting the catalog so you have to make a copy...
    mals_reordered = mals_catalog['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit', 'e_Spwfit', 'Scode', 'SPW/WB']
    fermi_reordered = Fermi_catalog['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng']

    '''FERMI OVERLAY'''
    # i would call in this into the catalog overlay not anything else
    mals_cataloging_uss = Catalog(mals_reordered, ['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit', 'e_Spwfit',  'Scode', 'SPW/WB'], catalog_type='uss')
    # mals_cataloging_css = Catalog(mals_reordered_2, ['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit'], catalog_type='css')
    fermi_cataloging = Catalog(fermi_reordered,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'], catalog_type='xray' )
    # since this is forced to be in a [] maybe fix it for single iteration if the user decides not to put it into a ls format?
    fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_cataloging, comparision_catalogs=[mals_cataloging_uss])

    fermi_ellipses_final, mal_cat_final, dict_mal_femi, uss_candidate_catalog  = fermi_mals_overlay.matched_catalogs()
    fermi_ellipses_final = fermi_ellipses_final[0]
    mal_cat_final = mal_cat_final[0]
    # ls all catalogs without the matching
    # dict key
    dict_mal_femi = dict_mal_femi[0]
    # dict_values = [*dict_mal_femi.values()]
    # dict_keys = list(dict_mal_femi.keys())
 
    '''SPW CROSSMATCHING'''
    spw2_catalog = mal_cat_final[mal_cat_final['SPW/WB'] == 'LSPW_2']
    spw9_catalog = mal_cat_final[mal_cat_final['SPW/WB'] == 'LSPW_9']

    spw2_cataloging = Catalog(spw2_catalog, ['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit', 'Scode', 'SPW/WB'])
    spw9_cataloging = Catalog(spw9_catalog, ['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit', 'Scode', 'SPW/WB'])
    spw2_spw9_overlay = CatalogOverlayer(base_catalog=spw2_cataloging, comparision_catalogs=[spw9_cataloging], beam_scale=2)

    # this will return only the matched sources while the mal_cat_final wil have the indexes of all sources,
    # so the indexes in mal_cat not in the spw2_9 match are the ones of interest 
    spw2_matched_catalog, spw9_matched_catalog, dict_spw2_sp9, spw9_all_candidates = spw2_spw9_overlay.matched_catalogs()
    spw2_matched_catalog = spw2_matched_catalog[0]
    spw9_matched_catalog = spw9_matched_catalog[0]
    dict_spw2_sp9 = dict_spw2_sp9[0]
    spw9_all_candidates = spw9_all_candidates[0]

    print(f'All Sources {len(mal_cat_final)}')
    print(f'Cross Spectral Window Matches {len(dict_spw2_sp9.keys())}')
    
    # combines nearby sources across spectral bands
    unique_sources_sp2 = spectral_window_merging(spw2_catalog, spw9_catalog, dict_spw2_sp9 )
    dict_mal_femi_global_index = global_index(dict_mal_femi, uss_candidate_catalog)
    dict_mal_femi_updated = {key: [value for value in value_list if value in (unique_sources_sp2['Indexes'])] 
                            for key, value_list in dict_mal_femi_global_index.items()
                             }
    dict_values = [*dict_mal_femi_updated.values()]
    dict_keys = list(dict_mal_femi_updated.keys())
    
    '''FERMI PLOTTING'''
    # defining the ellipse values
    # change this to a function in the future
    ra_fermi = coord.Angle(fermi_ellipses_final['RAJ2000']*u.deg)
    ra_fermi = - ra_fermi
    ra_reordered_fermi = ra_fermi.wrap_at(180*u.deg)
    dec_fermi = coord.Angle(fermi_ellipses_final['DEJ2000']*u.deg)
    semi_major_ax = fermi_ellipses_final['Conf_95_SemiMajor']*u.deg
    semi_minor_ax = fermi_ellipses_final['Conf_95_SemiMinor']*u.deg
    position_angle = fermi_ellipses_final['Conf_95_PosAng']*u.deg
    
    # makes fermi patches
    fermi_ellipses_patches = []
    for i in range(len(fermi_ellipses_final['RAJ2000'])):
        fermi_ellipses_patch = ellipse((ra_reordered_fermi[i].value, dec_fermi[i].value), semi_major_ax[i].value*2, semi_minor_ax[i].value*2, angle=position_angle[i].value, lw=2, edgecolor='b', alpha=0.5, zorder=1)
        fermi_ellipses_patches.append(fermi_ellipses_patch)
    # will save figure of a plt figure
    # plotting the first iteration of the ferm ellipses overlap THIS IS VARIABLE change if desired
    
    # ask for an input of how many sources lens to show max someone would want to change? print(input(len...))
    # print(f'ALL USS sources within Fermi Ellipses: {len(mal_cat_final)}')
    print(f'Unique USS sources within Fermi Ellipses: {len(unique_sources_sp2)}')
    print(f'Fermi Ellipses: {len(fermi_ellipses_final['RAJ2000'])}')
    
    print(fermi_ellipses_final)
    for i in range(len(fermi_ellipses_patches)):
        fermi_plot(fermi_ellipses_patches[i], dict_keys[i], unique_sources_sp2, dict_values[i])
        
            
if __name__ == '__main__':
    main()
