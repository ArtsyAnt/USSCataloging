from regions import EllipseSkyRegion
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from astropy.table import Table, Column, MaskedColumn
from astropy.table import vstack

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.io import fits
# from astropy.table import Table
"""Adding ID as a required parameter for the catalog so that it can be matched to the full catalog and we regain all the info we previously lost!"""
# idea for best guessing parameter call by doing a match case with some of its key words (but only accepts one column per parameter) 
# would start class out to be with this best guess feature in which: (first make all uppercase .upper then) (ra: RA), (dec: DEC), (semimaj: MAJ),
# (semiminor: MIN), (postion_angle: PA or POSANG or POSITIONANGLE) (ID: should just be any unique Identifier, ID, or SOURCE_NAME, NAME), 
# (the fluxes & spec IDX might be a bit harder!)

"""Todo:"""

"""1. come up with a better way to get the required column values -- a dict call"""
"""2. if calling from the same catalog make a copy to avoid error pop up between css and uss"""
"""3. add a print final pop up for the total # of matched indexs"""
"""4. create a new line for the output images so they dont go off onto the screen, maybe just make smaller if the len is larger?"""
# done - 6/12
# """5. add a INDEX column without having to call for it so that it can reconstruct base catalog with minimal effort, 
# I only call the index of the REDUCED CATALOG not really helpful in this case """  
"""maybe this means changing the init params to take the full catalog, and a dict of the required params so im not doing this silly list thing like ra:BLANK dec:BLANK"""
# done - 6/13
# """6. fill the catalog right when you get it!"""

# the point of this is just to reduce a catalog, right now
# only for simple masking/filling and the radio features
# but it can use the masking feature or filled feature to be a simple reduction
class Catalog:
    # assumes that it is already called from VizieR or is a table
    # assumes that the user reordered their columns in the order of :
    # ra, dec, semimajor, semiminor, flux, peak_flux, specindex (easy to reorder with astropy)
    def __init__(self, catalog, catalog_labels, catalog_type=None):
        # catalog type is between "radio", "infared", "optical" and "infared"
        # initalize:
        # catalog
        # catalog type
        try:
            self.catalog = catalog.filled(np.nan)
        except AttributeError:
            self.catalog = catalog
            
        self.type    = catalog_type
        self.inital_size = len(self.catalog)
        # call these from the initalized catalog 
        # these are only labels replace with dict
    
        self.ra  = catalog_labels[0]
        self.dec = catalog_labels[1]
        self.semimajor = catalog_labels[2]
        self.semiminor = catalog_labels[3]
        self.positionangle = catalog_labels[4]
        match self.type:
            case 'uss' | 'css':
                '''changed for scode temp'''
                # flux
                self.flux = catalog_labels[5]
                # peak_flux
                self.peak_flux = catalog_labels[6]
                # specindex
                self.spec_index = catalog_labels[7]  
                # specindex error
                self.spec_error = catalog_labels[8]
                # scode
                self.scode = catalog_labels[9]

            case 'fermi':
                ...
                
            case _:

                self.flux = None
                self.peak_flux = None
                self.spec_index = None
        
        # add indexes list
        if 'Indexes' in self.catalog.colnames:
            pass
        else:
            indexes = list(range(len(self.catalog)))
            self.catalog.add_column(indexes, name='Indexes')
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
        self.catalog = self.catalog
        return self.catalog
    # radio only
    # -----
    def radio_reduction(self, alpha_threshold=None):
        c_step_1 = self.catalog[self.catalog['FluxRatio'] <=1.1]
        c_step_2 = c_step_1[c_step_1['FluxRatio'] >=0.8]
        # testing s_code (the ngauss == 1 is the same)
        c_step_3 = c_step_2[c_step_2[self.scode] =='S']
        # ensure spectral index exists
        spec_idx_mask = np.isnan(c_step_3[self.spec_index])
        c_step_4 = c_step_3[~spec_idx_mask]
        # remove all spectral index errors that are crazy!
        c_step_5 = c_step_4[c_step_4[self.spec_error] >-999]

        # uss part
        match self.type:
            case 'uss':
                alpha_threshold = -1.4
                self.catalog = c_step_5[c_step_5[self.spec_index]< alpha_threshold]

            case 'css':
                alpha_threshold = -1
                self.catalog = c_step_5[c_step_5[self.spec_index]< alpha_threshold]

        print(f'Sources removed: {self.inital_size-len(self.catalog)}')
        print(f'{len(self.catalog)} {self.type} candidates')
        
# the point of this is that we can have mutiple different objects == differnt overlays
# a overlay ultimate goal is to create a list of indexes that match between the comparision
# catalogs with the base catalog

# we inherent the base catalogs ordered in the above way to make the calculations go quicker
# that part requires user inputclass CatalogPositionOverlay(Catalogs):
# calling base catalogs and comparision catalogs 
# the base catalog should be a radio source or something with a large area
# for two catalogs only takes about ~2 minutes (with ~200k & 7195 sources)
class CatalogOverlayer:
    def __init__(self, base_catalog: Catalog, comparision_catalogs: Catalog, **kwargs): 
        self.base = base_catalog
        self.comparision = comparision_catalogs
        # the number of catalogs being compared 
        '''rename'''
        self.comparision_index = len(self.comparision)
        # size scaling for capturing within the regions area
        self.size_scale = kwargs.get('beam_scale', 1)
    
        # could make a attribute that follows the number of comparision catalogs
        for index in range(self.comparision_index):
            comparing_cat = self.comparision[index]
            # in here we would then just call for the general dict reduction and it will be further curated by the keys in the dict
            # change to is in keys in future
            match comparing_cat.type:
                case 'uss':
                    comparing_cat.flux_ratio()
                    comparing_cat.radio_reduction()
            
                case 'css':
                    comparing_cat.flux_ratio()
                    comparing_cat.radio_reduction()
                case _:
                    comparing_cat.robust()
                    
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
        # (make into a four loop?)
        ra_center_ellipses  = self.base.catalog[self.base.ra]
        dec_center_ellipses = self.base.catalog[self.base.dec]
        # maybe have to put .unit()
        try: 
            semi_major_axis     = self.base.catalog[self.base.semimajor].to(u.deg)*self.size_scale
            semi_minor_axis     = self.base.catalog[self.base.semiminor].to(u.deg)*self.size_scale
        except:
            semi_major_axis     = self.base.catalog[self.base.semimajor]*u.deg*self.size_scale
            semi_minor_axis     = self.base.catalog[self.base.semiminor]*u.deg*self.size_scale
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
                radius = (x_rotated**2/(semi_major_axis[iteration])**2)+ ((y_rotated**2)/(semi_minor_axis[iteration])**2)       
                
                # 6. append radius of all mals sources to fermi ellipse [i]
                self.comparision[index].radiuses_from_ellipse.append(radius)
                self.comparision[index].is_in_list.append((radius<=1))
                # why would any one realistically want just the boolean
        # does it need to return anything?
        return  self.comparision

    # creates a dict of the ellipse, pointing relationship, and a index for all the matched points
    # does this for all compared catalogs
    def ellipse_point_indexing(self):
        # self.comparision_index and index is hard to read!
        for index in range(self.comparision_index):
            comparing_cat = self.comparision[index]
            
            major_indices = [ellipse_index for ellipse_index,
                             ellipse in enumerate(comparing_cat.is_in_list) if any(ellipse)]
            
            # I have a radiuses array should I include it in output?
            # radiuses_array = np.array(radiuses_from_ellipse)[major_indices]
            
            # array of the ellipses that have atleast one point within it.
            filtered_list = np.array(comparing_cat.is_in_list)[major_indices]
            
            # change later there are better sub filter algos
            # basic algo. to sort the ellipse index value in fermi catalog to the index of the point
            self.comparision[index].ellipse_point_index = {}
            self.comparision[index].point_all_indexes   = []
            self.comparision[index].ellipse_indexes     = major_indices
            j=0
            for ellipse in filtered_list:
                point_indices = []
                index_num = 0 
                # iterates to see if the index number is in the ellipse 
                # Im going to change it so that it will be linked to the actual inital index so that no matter what the indexes can match
                # that way the indexes that are held onto can correlate across similar catalogs easier!
                for indice in ellipse:
                    if indice == True:
                        # this will input the inital indexes to transfer across catalogs!
                        # point_indices.append(comparing_cat.catalog['Indexes'][index_num])
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
        all_matched_ellipses = []
        ellipse_point_dicts  = []
        reduced_compared_cats = []
        
        # get indexes and reduces catalogs to only be the matching ones 
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

'''UPDATE TO CONSIDER THE INDEXES FROM THE BASE CATALOG!'''
def fermi_plot(ellipse, ellipse_indexes, point_catalog, point_indexes):
    # can add the CSS_indexes later
    fig = plt.figure(figsize=(12,12))
    ax3 = plt.gca()

    new_ellipse = copy(ellipse)
    # get the ra and dec of the mal catalog
    # start the list here

    # capture either iteration method!
    try:
        point_catalog = point_catalog[point_indexes]
    except:
    # it will iterate through all of them in the fermi plotting ?
    # since these are unique per ellipse it wont be an issue?
    # this should work even if I am not doing the mals spectral combining!
        point_catalog = point_catalog[np.isin(point_catalog['Indexes'], point_indexes)]
    ra = point_catalog['RAJ2000']
    dec = point_catalog['DEJ2000']
    
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

def spectral_window_merging(base_window, overlapping_window, matched_dict):
  # indicies seperation
  base_window_new = base_window.copy()
  keys_list = list(matched_dict.keys())
  comparision_spw_indices = []
  for spw_index in keys_list:
      comparision_spw_indices.append(matched_dict[spw_index][0])
      
  # call any index really
  Spectral_window_type = overlapping_window['SPW/WB'][0]
  column_length        = len(base_window)
  empty_data           = np.zeros(column_length)
  
  # all the empty columns
  flux      = MaskedColumn(empty_data, name = f'Flux_{Spectral_window_type}').filled(np.nan)
  Peak_flux = MaskedColumn(empty_data, name = f'FluxPk_{Spectral_window_type}')
  Spw_fit   = MaskedColumn(empty_data, name = f'Spwfit_{Spectral_window_type}')
  Spw_efit  = MaskedColumn(empty_data, name = f'e_Spwfit_{Spectral_window_type}')

  #   # get overlap for all indexes
  #   overlap = np.zeros(len(base_window)+len(overlapping_window), dtype=bool)
  #   overlap[comparision_spw_indices] = True
      
  mask = np.ones(len(overlapping_window), dtype=bool)
  mask[comparision_spw_indices] = False
  unique_sources = overlapping_window[mask] 
  for index in range(len(keys_list)):
      flux[keys_list[index]] = overlapping_window['Flux'][comparision_spw_indices[index]]
      Peak_flux[keys_list[index]] = overlapping_window['FluxPk'][comparision_spw_indices[index]]
      Spw_fit[keys_list[index]] = overlapping_window['Spwfit'][comparision_spw_indices[index]]
      Spw_efit[keys_list[index]] = overlapping_window['e_Spwfit'][comparision_spw_indices[index]]
# for some reason failed when i assigned it to a catalog, catalog calling issue.
  try:
      base_window_new.add_columns([flux,Peak_flux,Spw_fit,Spw_efit])
  except:
      base_window_new = base_window_new
    #   print(f'Unique source catalog {unique_sources} ')
  unique_detections = vstack([base_window_new, unique_sources], join_type='outer')
  # unique_detections.add_column(overlap, name='Spw_overlap')
  return unique_detections


# I will make a function that will always convert the individual indexing to a 'global' indexing'!
# just uses the dict comprehension, this allows the indexes to act as
# connecting point between steps in catalog reduction if there are intermediary steps in mutiple catalog calls for same base!
def global_index(dict, dict_catalog):
    dict_global = {key: [dict_catalog['Indexes'][value] for value in value_list] 
                   for key, value_list in dict.items()}
    return dict_global
