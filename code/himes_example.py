# example of Fermi Unassociated Sources in the MeerKAT Absorption Line Survey catalog reduction using classes

from multiwavecataloging import Catalog
from multiwavecataloging import CatalogOverlayer
from multiwavecataloging import fermi_plot

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.io import fits
from astropy.table import vstack
from astropy.table import Table
from astropy import wcs
from regions import EllipseSkyRegion

# Access astronomical databases
from astroquery.vizier import Vizier

# For plots
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as ellipse

# Data handling
import numpy as np
import specutils.analysis as spec_ana
from scipy.optimize import curve_fit
from specutils import SpectralRegion, Spectrum1D


def main():    
    # catalog call
    vizier = Vizier(columns=[])
    vizier.ROW_LIMIT = -1# -1 means unlimited
    
    mals_catalog = vizier.get_catalogs('J/ApJS/270/33')[0]
    Fermi_hud = fits.open('/Users/mario/Coding/nrao_reu_research/socorro/USSCataloging/catalogs/gll_psc_v35.fit')
    Fermi_catalog = Table(Fermi_hud[1].data)
    
    # reordering 
    # weird its double counting the catalog so you have to make a copy...
    mals_reordered = mals_catalog['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit']
    # mals_reordered_2 = mals_catalog['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit']
    fermi_reordered = Fermi_catalog['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng']


    # i would call in this into the catalog overlay not anything else
    mals_cataloging_uss = Catalog(mals_reordered, ['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit'], catalog_type='uss')
    # mals_cataloging_css = Catalog(mals_reordered_2, ['RAJ2000', 'DEJ2000', 'Majaxis', 'Minaxis', 'PA', 'Flux', 'FluxPk', 'Spwfit'], catalog_type='css')
    fermi_cataloging = Catalog(fermi_reordered,['RAJ2000', 'DEJ2000','Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng'], catalog_type='xray' )
    # since this is forced to be in a [] maybe fix it for single iteration if the user decides not to put it into a ls format?
    fermi_mals_overlay = CatalogOverlayer(base_catalog=fermi_cataloging, comparision_catalogs=[mals_cataloging_uss])

    # this is a temp solution, we would rather have the indexing values be the catalog's 'ID' values 
    # so that we dont have to revert back to calling the semi reduced catalog (uss-canidate in this case) rather than just call 
    # the exampled mal_cat_final. Also would clear clutter.
    fermi_ellipses_final, mal_cat_final, dict_mal_femi, uss_candidate_catalog  = fermi_mals_overlay.matched_catalogs()

    fermi_ellipses_final = fermi_ellipses_final[0]
    
    mal_cat_final = mal_cat_final[0]
    # ls all catalogs without the matching
    # dict key
    dict_mal_femi = dict_mal_femi[0]
    dict_values = [*dict_mal_femi.values()]
    dict_keys = list(dict_mal_femi.keys())

    # defining the ellipse values
    # change this to a function in the future
    ra_fermi = coord.Angle(fermi_ellipses_final['RAJ2000']*u.deg)
    ra_fermi = - ra_fermi
    ra_reordered_fermi = ra_fermi.wrap_at(180*u.deg)
    dec_fermi = coord.Angle(fermi_ellipses_final['DEJ2000']*u.deg)

    
    semi_major_ax = fermi_ellipses_final['Conf_95_SemiMajor']*u.deg
    semi_minor_ax = fermi_ellipses_final['Conf_95_SemiMinor']*u.deg
    position_angle = fermi_ellipses_final['Conf_95_PosAng']*u.deg
    
    fermi_ellipses_patches = []
    for i in range(len(fermi_ellipses_final['RAJ2000'])):
        fermi_ellipses_patch = ellipse((ra_reordered_fermi[i].value, dec_fermi[i].value), semi_major_ax[i].value*2, semi_minor_ax[i].value*2, angle=position_angle[i].value, lw=2, edgecolor='b', alpha=0.5, zorder=1)
        fermi_ellipses_patches.append(fermi_ellipses_patch)
    # will save figure of a plt figure
    # plotting the first iteration of the ferm ellipses overlap THIS IS VARIABLE change if desired
    
    # input the lens to show max someone would want to change? print(input(len...))
    fermi_plot(fermi_ellipses_patches[0], dict_keys[0], uss_candidate_catalog, dict_values[0])
        
if __name__ == '__main__':
    main()
