from regions import EllipseSkyRegion
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.io import fits
# from astropy.table import Table

"""Todo: come up with a better way to get the required column values"""
"""2. if calling from the same catalog make a copy to avoid error pop up between css and uss"""
"""3. add a print final pop up for the # of matched indexs"""
# the point of this is just to reduce a catalog, right now
# only for simple masking/filling and the radio features
# but it can use the masking feature or filled feature to be a simple reduction
class Catalog:
    # assumes that it is already called from VizieR or is a table
    # assumes that the user reordered their columns in the order of :
    # ra, dec, semimajor, semiminor, flux, peak_flux, specindex (easy to reorder with astropy)
    def __init__(self, catalog, catalog_labels, catalog_type):
        # catalog type is between "radio", "infared", "optical" and "infared"
        # initalize:
        # catalog
        # catalog type
        self.catalog = catalog
        # no current utility 
        self.type    = catalog_type
        # call these from the initalized catalog 
        # these are only labels  
        self.ra  = catalog_labels[0]
        self.dec = catalog_labels[1]
        self.semimajor = catalog_labels[2]
        self.semiminor = catalog_labels[3]
        self.positionangle = catalog_labels[4]
                
        match self.type:
            case 'uss':
                # flux
                self.flux = catalog_labels[5]
                # peak_flux
                self.peak_flux = catalog_labels[6]
                # specindex
                self.spec_index = catalog_labels[7]
            case 'css':
                # flux
                self.flux = catalog_labels[5]
                # peak_flux
                self.peak_flux = catalog_labels[6]
                # specindex
                self.spec_index = catalog_labels[7]
            case _:
                self.flux = None
                self.peak_flux = None
                self.spec_index = None
        
    # have a intermediary reduction step?
    
    # flux ratio 
    def flux_ratio(self):
        self.flux_ratio_ = self.catalog[self.peak_flux]/self.catalog[self.flux]
        self.catalog.add_column(self.flux_ratio_, name='FluxRatio')
    # baseline for all sources is that it is a compact source
    # and that none of the columns have empty spots 
    
    # any catalog type
    # ---
    def robust(self):
        self.catalog = self.catalog.filled(np.nan)
        return self.catalog
    
    # radio only
    # -----
    def uss(self, alpha_threshold=-1.4):
        c_step_1 = self.catalog[self.catalog['FluxRatio'] <=1.2]
        c_step_2 = c_step_1[c_step_1['FluxRatio'] >=0.8]
        # ensure spectral index exists
        spec_idx_mask = np.isnan(c_step_2[self.spec_index].filled(np.nan))
        c_step_3 = c_step_2[~spec_idx_mask]
        # uss part
        self.uss_sources = c_step_3[c_step_3[self.spec_index]< alpha_threshold]

        print(f'Sources removed: {len(self.catalog)-len(self.uss_sources)}')
        print(f'{len(self.uss_sources)} USS candidates')
        # overwrite the catalog
        self.catalog = self.uss_sources.filled(0)
        return self.catalog 

    def css(self, alpha_threshold=-1):
        c_step_1 = self.catalog[self.catalog['FluxRatio'] <=1.2]
        c_step_2 = c_step_1[c_step_1['FluxRatio'] >=0.8]
        # ensure spectral index exists
        spec_idx_mask = np.isnan(c_step_2[self.spec_index].filled(np.nan))
        c_step_3 = c_step_2[~spec_idx_mask]
        # css part
        self.css_sources = c_step_3[c_step_3[self.spec_index]< alpha_threshold]
        # input angular size limiter

        print(f'Sources removed: {len(self.catalog)-len(self.css_sources)}')
        print(f'{len(self.css_sources)} USS candidates')
        # overwrite catalog
        self.catalog = self.css_sources.filled(0)
        return self.catalog 
        
# the point of this is that we can have mutiple different objects == differnt overlays
# a overlay ultimate goal is to create a list of indexes that match between the comparision
# catalogs with the base catalog

# we inherent the base catalogs ordered in the above way to make the calculations go quicker
# that part requires user inputclass CatalogPositionOverlay(Catalogs):
# calling base catalogs and comparision catalogs 
# the base catalog should be a radio source or something with a high end to end emission
# while fermi error ellipises 
class CatalogOverlayer:
    def __init__(self, base_catalog: Catalog, comparision_catalogs: Catalog): 
        self.base = base_catalog
        self.comparision = comparision_catalogs
        # the number of catalogs being compared
        self.comparision_index = len(self.comparision)
    
        # could make a attribute that follows the number of comparision catalogs
        for index in range(self.comparision_index):
            comparing_cat = self.comparision[index]
            if comparing_cat.type == 'uss':
                comparing_cat.flux_ratio()
                comparing_cat.uss()
            
            elif comparing_cat.type == 'css':
                comparing_cat.flux_ratio()
                comparing_cat.css()
        
        # this might take like 5+ min just for 2 cat and a base one?
        # check if in ellipse
        self.is_in_ellipse()
        # redefine ellipse into dict and other forms 
        self.ellipse_point_indexing()

    # only need if the coordinate systems dont align, this should be a check to add in the ra and dec key word search
    def coordinate_matching(self):
        ...
    # this is good for only one pairing 
    def is_in_ellipse(self):
        # only for ones who have the same coordinate system so be wary of that
        # a more friendly way to do this would be to make it into a dict so
        # there is less of an error in order
        # 1. decompose points
        # kind of disgusting redo it (make into a four loop?)
        ra_center_ellipses  = self.base.catalog[self.base.ra]
        dec_center_ellipses = self.base.catalog[self.base.dec]
        semi_major_axis     = self.base.catalog[self.base.semimajor]
        semi_minor_axis     = self.base.catalog[self.base.semiminor]
        position_angle      = self.base.catalog[self.base.positionangle]
        
        for index in range(self.comparision_index):
            # puts the is_in_list and radiuses array in each compared cat for later
            self.comparision[index].is_in_list=[]
            self.comparision[index].radiuses_from_ellipse = []
            comparing_cat = self.comparision[index]
            # arbritrary call just for a indexing in the base cat change to size?
            for iteration in range(len(self.base.catalog[self.base.ra])):
                
                ra_points = comparing_cat.catalog[comparing_cat.ra]
                dec_points = comparing_cat.catalog[comparing_cat.dec]
                
                # 2. defining ra as x and dec as y
                x_distances_from_center = ra_points  - ra_center_ellipses[iteration]*u.deg
                y_distances_from_center = dec_points - dec_center_ellipses[iteration]*u.deg
                
                # 3. define angles since we are using it reverse should be reversed
                theta = np.radians(position_angle[iteration]*u.deg)
                cos = np.cos(theta)
                sin = np.sin(theta)
                
                # 4. get rotated x and y 
                x_rotated = x_distances_from_center * cos - y_distances_from_center *sin
                y_rotated = x_distances_from_center * sin + y_distances_from_center *cos
                    
                # 5.check if and y_rot are in the ellipse equation and make a append
                radius = (x_rotated**2/(semi_major_axis[iteration]*u.deg)**2)+ ((y_rotated**2)/(semi_minor_axis[iteration]*u.deg)**2)       
                
                # 6. append radius of all mals sources to fermi ellipse [i]
                self.comparision[index].radiuses_from_ellipse.append(radius)
                self.comparision[index].is_in_list.append((radius<=1))
                # why would any one realistically want just the boolean
        # does it need to return anything?
        return  self.comparision

    # creates a dict of the ellipse, pointing relationship, and a index for all the matched points
    # does this for all compared catalogs
    def ellipse_point_indexing(self):
        for index in range(self.comparision_index):
            comparing_cat = self.comparision[index]
            
            major_indices = [ellipse_index for ellipse_index,
                             ellipse in enumerate(comparing_cat.is_in_list) if any(ellipse)]
            
            # have a radiuses array include it?
            # radiuses_array = np.array(radiuses_from_ellipse)[major_indices]
            filtered_list = np.array(comparing_cat.is_in_list)[major_indices]
            
            # change later there are better sub filter algos
            self.comparision[index].ellipse_point_index = {}
            self.comparision[index].point_all_indexes   = []
            self.comparision[index].ellipse_indexes     = major_indices
            j=0
            for ellipse in filtered_list:
                point_indices = []
                index_num = 0 
                for indice in ellipse:
                    if indice == True:
                        point_indices.append(index_num)
                        self.comparision[index].point_all_indexes.append(index_num)
                        index_num +=1
                    else:
                        index_num +=1
                self.comparision[index].ellipse_point_index[major_indices[j]] = point_indices
                j+=1
        # return self.comparision

# you asked for the matched catalogs back >:)
    def matched_catalogs(self):
        # this is the issue it cant be a set the order is unique fix tomorrow
        all_matched_ellipses = []
        ellipse_point_dicts  = []
        reduced_compared_cats = []
        
        # get indexes and reduces catalogs 
        for index in range(self.comparision_index):
            ellipse_indexes = self.comparision[index].ellipse_indexes
            point_indexes   =  self.comparision[index].point_all_indexes
            ellipse_point_dict = self.comparision[index].ellipse_point_index
            
            reduced_compared_cat = self.comparision[index].catalog[point_indexes]
            reduced_compared_cats.append(reduced_compared_cat)
            
            ellipse_point_dicts.append(ellipse_point_dict)
            all_matched_ellipses.append(ellipse_indexes)
            
        # now append_reduced catalogs
        self.base_matched_catalog        = self.base.catalog[all_matched_ellipses]
        self.comparision_matched_catalog = reduced_compared_cats
        self.dict_match                  = ellipse_point_dicts
        
        # returns: matched_base_catalog, [a single list?]
        # matched compared cat [a list]
        # dict of all ellipse - idx per catalog [a list]
        # and the first catalog without the matching THIS IS TEMP
        return self.base_matched_catalog, self.comparision_matched_catalog, self.dict_match, self.comparision[0].catalog
    
    
# something to visualize the catalog overlay in different ways?
class Multicatalogvisual():
    ...
# make a function that will plt the values of the scatter in that region and the patch
def fermi_plot(ellipse, ellipse_indexes, point_catalog, point_indexes):
    # can add the CSS_indexes later
    fig = plt.figure(figsize=(12,12))
    ax3 = plt.gca()

    new_ellipse = copy(ellipse)
    # get the ra and dec of the mal catalog
    # start the list here
    ra = point_catalog[point_indexes]['RAJ2000']
    dec = point_catalog[point_indexes]['DEJ2000']
    
    # ra reordering
    ra_coords = coord.Angle(ra)
    ra_coords = -ra_coords
    ra_reordered_coords = ra_coords.wrap_at(180*u.deg)
    dec_coords = coord.Angle(dec)

    # get the width and height
    semi_major = new_ellipse.get_width()/2
    semi_minor = new_ellipse.get_height()/2
    
    # get the x and y from the ellipse center and then add x degree on both side and up down
    ra_ellipse, dec_ellipse = new_ellipse.get_center()
    
    # put the area in and show it 
    area = np.pi * semi_major * semi_minor
    
    # plot the ellipse
    ax3.add_patch(new_ellipse)
    
    labels = point_indexes
    # for index in len(range())
    plt.scatter(ra_reordered_coords, dec_coords, s=5, color='green', edgecolor='black', marker='.', zorder = 2, label = f'USS Indexes: {(labels)}' )
    plt.legend(edgecolor='black')
    
    # arbitrary size limit 
    plt.xlim(ra_ellipse-1, ra_ellipse+1)
    plt.ylim(dec_ellipse-1, dec_ellipse+1)

    # uss scatter

    # plt area 
    anchored_text = AnchoredText(f'Ellipse Area: {area} \nID: {ellipse_indexes}', loc =2)
    ax3.add_artist(anchored_text)
    
    plt.xlabel('RAJ2000 [deg]')
    plt.ylabel('DEJ2000 [deg]')

    plt.savefig('fermi_ellipse_example.jpg', dpi=300)
    plt.show()
    # change x ticks to degs or hours