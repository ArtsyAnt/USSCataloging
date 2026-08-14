from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Ellipse as ellipse_mals

from astropy.table import Table, Column, MaskedColumn
from astropy.table import vstack

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import search_around_sky
import astropy.coordinates as coord
from astropy.io import fits

"""Adding ID as a required parameter for the catalog so that it can be matched to the full catalog and we regain all the info we previously lost!"""
# idea for best guessing parameter call by doing a match case with some of its key words (but only accepts one column per parameter) 
# would start class out to be with this best guess feature in which: (first make all uppercase .upper then) (ra: RA), (dec: DEC), (semimaj: MAJ),
# (semiminor: MIN), (postion_angle: PA or POSANG or POSITIONANGLE) (ID: should just be any unique Identifier, ID, or SOURCE_NAME, NAME), 
# (the fluxes & spec IDX might be a bit harder!)

"""Todo:"""
"""1. come up with a better way to get the required column values -- a dict call"""
# done - 6/22 - adding .copy()
# """2. if calling from the same catalog make a copy to avoid error pop up between css and uss"""
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
    def __init__(self, catalog, catalog_labels,catalog_name =None, catalog_type=None):
        # catalog type is between "radio", "infared", "optical" and "infared"
        # initalize:
        # catalog
        # catalog type
        try:
            # changed this night of 6/29 if there is a random issue just flip order!
            # self.catalog = catalog.copy()
            self.catalog = catalog.filled(np.nan)
        except TypeError:
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
            case 'uss' | 'css' | 'radio':
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

            case 'fermi' |'gamma' |'xray':
                ...
                
            case _:
                self.flux = catalog_labels[5]
                # peak_flux
                self.peak_flux = catalog_labels[6]
                # specindex
                self.spec_index = catalog_labels[7]  
                self.catalog[self.flux] = len(self.catalog) * [self.flux]
                self.catalog[self.semimajor] = len(self.catalog) * [self.semimajor]
                
        # add indexes list
        if 'Indexes' in self.catalog.colnames:
            pass
        else:
            indexes = list(range(len(self.catalog)))
            self.catalog.add_column(indexes, name='Indexes')
            
        if catalog_name == None:
            pass
        else:
            catalog_size = len(self.catalog)
            name_column= catalog_size* [catalog_name]
            self.catalog['catalog_names'] = name_column
    # have a intermediary reduction step?
    # flux ratio 
    def flux_ratio(self):
        self.flux_ratio_ = self.catalog[self.peak_flux].value/self.catalog[self.flux].value
        self.catalog.add_column(self.flux_ratio_, name='FluxRatio')
    # baseline for all sources is that it is a compact source
    # and that none of the columns have empty spots 
    # any catalog type
    # ---
    # dont know if I need it?
    def robust(self):
        # if no units are given input the units we expect for these classes to run
        self.catalog[self.ra]  = self.catalog[self.ra]*u.deg
        self.catalog[self.dec] = self.catalog[self.dec]*u.deg
        self.catalog[self.semimajor] = self.catalog[self.semimajor]*u.deg
        self.catalog[self.semiminor] = self.catalog[self.semiminor]*u.deg
        self.catalog[self.positionangle] =self.catalog[self.positionangle]*u.deg
        self.catalog[self.flux] =self.catalog[self.flux] *u.mJy
        self.catalog[self.peak_flux] =self.catalog[self.peak_flux] *u.mJy/u.beam
        
        self.catalog = self.catalog
        return self.catalog
    # radio only
    # -----
    def radio_reduction(self, alpha_threshold=None, himes_instance=False, **kwargs):
        self.type=kwargs.get('catalog_type', self.type)
        if himes_instance == True:
            c_step_1 = self.catalog[self.catalog['FluxRatio'] <=1.2]
            # testing s_code (the ngauss == 1 is the same)
            c_step_3 = c_step_1[c_step_1[self.scode] =='S']
        else:
            c_step_1 = self.catalog[self.catalog['FluxRatio'] <=1.1]
            c_step_2 = c_step_1[c_step_1['FluxRatio'] >=0.8]
            # testing s_code (the ngauss == 1 is the same)
            c_step_3 = c_step_2[c_step_2[self.scode] =='S']
        # ensure spectral index exists
        # consider not just removing
        spec_idx_mask = np.isnan(c_step_3[self.spec_index])
        c_step_4 = c_step_3[~spec_idx_mask]
        # remove all spectral index errors that are crazy!
        # not just errors from older consider other things
        c_step_5 = c_step_4[c_step_4[self.spec_error] >-999]

        if alpha_threshold != None:
            self.catalog = c_step_5[c_step_5[self.spec_index] < alpha_threshold]
            return self 

        else:
            # uss part
            match self.type:
                case 'uss':
                    alpha_threshold = -1.4
                    self.catalog = c_step_5[c_step_5[self.spec_index]< alpha_threshold]
                    return self 


                case 'css':
                    alpha_threshold = -1
                    self.catalog = c_step_5[c_step_5[self.spec_index]< alpha_threshold]
                    return self 

        print(f'Sources removed: {self.inital_size-len(self.catalog)}')
        print(f'{len(self.catalog)} {self.type} candidates')
                
# the point of this is that we can have mutiple different objects == differnt overlays
# a overlay ultimate goal is to create a list of indexes that match between the comparision
# catalogs with the base catalog

# we inherent the base catalogs ordered in the above way to make the calculations go quicker
# that part requires user inputclass CatalogPositionOverlay(Catalogs):
# calling base catalogs and comparision catalogs 
# the base catalog should be a radio source beam or something with a large error area
# for two catalogs only takes about ~30 minutes (with ~200k & 7195 sources) for matching 
class CatalogOverlayer:
    def __init__(self, base_catalog: Catalog, comparision_catalogs: Catalog, **kwargs): 
        self.base = base_catalog
        self.comparision = comparision_catalogs
        # the number of catalogs being compared 
        '''rename'''
        self.comparision_index = len(self.comparision)
        print(self.comparision_index )
        # size scaling for capturing within the regions area
        self.size_scale = kwargs.get('beam_scale', 1)
        # add ang res
        self.add_ang_sep = kwargs.get('add_ang_sep', False)

    def is_in_ellipse_V2(self):
        # circle-reduction (k-d tree via search_around_sky) narrows candidates to
        # everything within the largest base ellipse's semi-major axis, then the
        # exact rotated-ellipse equation is evaluated on every candidate pair at
        # once (no per-ellipse Python loop) before grouping back into a dict.
        try:
            self.base.semi_major_ax = self.base.catalog[self.base.semimajor].to(u.deg)*self.size_scale
            self.base.semi_minor_ax = self.base.catalog[self.base.semiminor].to(u.deg)*self.size_scale
        except:
            self.base.semi_major_ax = self.base.catalog[self.base.semimajor]
            self.base.semi_minor_ax = self.base.catalog[self.base.semiminor]
        try:
            self.base.position_angle = self.base.catalog[self.base.positionangle]
        except:
            #for N/A
            if self.base.positionangle == 'N/A':
                self.base.position_angle = [0] * len(self.base.catalog)
        # take largest sep. limit from all the ellipses and use it as a catch all range
        largest_sep_limit = max((self.base.semi_major_ax).value)
        for index in range(self.comparision_index):
            print(f'===={index+1} catalog overlay====')
            comparing_cat = self.comparision[index]
            # the inital masking (with skycoord uses k-tree matching so faster?!
            robust_comparision_coords = SkyCoord(ra=comparing_cat.catalog[comparing_cat.ra], dec=comparing_cat.catalog[comparing_cat.dec])
            robust_base_coords = SkyCoord(ra=self.base.catalog[self.base.ra], dec=self.base.catalog[self.base.dec])
            idx_tar, idx_catal, ang_seperation, _ = search_around_sky(robust_comparision_coords, robust_base_coords, seplimit=largest_sep_limit*u.deg)
            if len(idx_catal) == 0:
                print('no matches')
                continue
                
            print('circle-reduction')
            print(f'Remaining point matches {len(comparing_cat.catalog[idx_tar])}')
            print(f'Remaining ellipses {len(self.base.catalog[list(set(idx_catal))])}')

            # adds ang res so I dont have to calc it again!
            if self.add_ang_sep==True:
                self.comparision[index].catalog['ang_sep'] = np.nan
                self.comparision[index].catalog['ang_sep'][idx_tar] = ang_seperation

            ra_ellipses  = self.base.catalog[self.base.ra]
            dec_ellipses = self.base.catalog[self.base.dec]

            # Evaluate the rotated-ellipse equation for every circle-reduced
            # candidate pair in one vectorized pass (fancy-indexed by idx_catal /
            # idx_tar) instead of looping per base ellipse.
            ra_points  = comparing_cat.catalog[comparing_cat.ra][idx_tar]
            dec_points = comparing_cat.catalog[comparing_cat.dec][idx_tar]
            x_distances_from_center = ra_points  - ra_ellipses[idx_catal]
            y_distances_from_center = dec_points - dec_ellipses[idx_catal]

            theta = np.radians(self.base.position_angle[idx_catal])
            cos = np.cos(theta)
            sin = np.sin(theta)
            x_rotated = x_distances_from_center * cos - y_distances_from_center * sin
            y_rotated = x_distances_from_center * sin + y_distances_from_center * cos

            radius = (x_rotated**2 / (self.base.semi_major_ax[idx_catal])**2) \
                   + (y_rotated**2 / (self.base.semi_minor_ax[idx_catal])**2)

            within_ellipse = radius <= 1
            kept_keys   = np.asarray(idx_catal)[within_ellipse]
            kept_values = np.asarray(idx_tar)[within_ellipse]

            # Group kept (base_idx -> [comparison_idx, ...]) via sort + split
            # instead of a per-pair Python dict-append loop.
            ellipses_dict = {}
            if len(kept_keys) > 0:
                order = np.argsort(kept_keys, kind='stable')
                sorted_keys = kept_keys[order]
                sorted_vals = kept_values[order]
                unique_keys, group_starts = np.unique(sorted_keys, return_index=True)
                grouped_vals = np.split(sorted_vals, group_starts[1:])
                ellipses_dict = {int(key): list(group) for key, group in zip(unique_keys, grouped_vals)}

            self.comparision[index].ellipse_point_index = ellipses_dict
            # at this point I am getting the indexes of the catalogs for the fermi
        #  the comparision class instances
        # should it return just the dictionaries since that is what it is doing
        return  self.comparision
        
# you asked for the matched catalogs back >:)
# will only return the matched catalogs unless you ask for the 'parent' catalogs for base and comparision to do wider matching with the indexes 
    def matched_catalogs_V2(self, return_parent_catalogs=False):
        ellipse_point_dicts        = []
        reduced_compared_cats      = []
        unreduced_comparision_cats = []
        base_catalogs              = []
         # get indexes and reduces catalogs to only be the matching ones 
        for index in range(self.comparision_index):
            # pull the matches of just the ellipses/keys
            ellipses= list(self.comparision[index].ellipse_point_index.keys())
            # pull the matches of all the points for comp catalogs
            point_indexes = [*self.comparision[index].ellipse_point_index.values()]
            point_indexes = [index for index_list in point_indexes for index in index_list]
            # the individual dictonaries per iteration
            # print(self.comparision[index].ellipse_point_index)
            
            # since it is a local reduction the index is a local positional match for the comparisions cats.
            reduced_compared_cat = self.comparision[index].catalog[point_indexes]
            reduced_compared_cats.append(reduced_compared_cat)
            ellipse_point_dict = self.comparision[index].ellipse_point_index
            ellipse_point_dicts.append(ellipse_point_dict)
            # append all the unassoc ellipses(before doing the reduction, so you can properly index into them across catalogs)
            
            base_catalogs.append(self.base.catalog[ellipses])
            unreduced_comparision_cats.append(self.comparision[index].catalog)
            
        
        # all the catalogs the dicts were generated from
            
        # now append_reduced catalogs
        self.base_matched_catalogs       = base_catalogs
        self.comparision_matched_catalog = reduced_compared_cats
        self.dict_match                  = ellipse_point_dicts
        
        # returns: matched_base_catalogs, [a list of tables, one per each comparision] that are only the matching ellipses
        # matched compared cat [a list]
        # dict of all ellipse - idx per catalog [a list]
        # and the first catalog without the matching THIS IS TEMP
        if return_parent_catalogs == False:
            return self.base_matched_catalogs, self.comparision_matched_catalog, self.dict_match
        else:
            comparision_cat = []
            for index in range(self.comparision_index):
                comparision_cat.append(self.comparision[index].catalog)
                # check if this ruins anythin
            self.comparision_catalogs = comparision_cat
            return self.base_matched_catalogs, self.comparision_matched_catalog, self.dict_match, self.base.catalog, self.comparision_catalogs
    

# '''UPDATE TO CONSIDER THE INDEXES FROM THEIR BASE CATALOG!'''
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
    # change x ticks to have ^degs or hours

# combines a base spectral window source with any other number of spectral windows given that the dictionary call was base to all
def spectral_window_merging_V2(base_window, overlapping_windows, matched_dict): 
    # is hardcoding the labels from the full mals cat
    # still dont get this function issues as much as I should 
    # # indicies seperation
    # get the base catalog and the spectral_window_of it and add the flux, fluxerr and freq for the base!
    base_window_new = base_window.copy()
    base_spwid = list(set(base_window_new['spw_id']))[0]
    
    base_window_new[f'total_flux_{base_spwid}'] = base_window_new['total_flux']
    base_window_new[f'total_flux_e_{base_spwid}'] = base_window_new['total_flux_e']
    base_window_new[f'ref_freq_{base_spwid}'] = base_window_new['ref_freq']

    # merge all the overlapping window catalogs 
    keys_list = list(matched_dict.keys())
    # this gets all of the indicies 
    comparision_spw_indices = []
    for spw_index in keys_list:
        # worried about this [0]
        comparision_spw_indices.append(matched_dict[spw_index][0])
        
    # take in all the spectral windows names for the overlapping windows 
    Spectral_window_types = list(set(overlapping_windows['spw_id']))
    column_length        = len(base_window_new)
    empty_data           = np.zeros(column_length)

    # add empty/nan columns for all the expected windows into the base window
    # get the units of the overlaping_window
    flux_unit = overlapping_windows['total_flux'].unit
    flux_err_unit = overlapping_windows['total_flux_e'].unit
    freq_unit = overlapping_windows['ref_freq'].unit


    for i in range(len(Spectral_window_types)):
        flux      = Column(empty_data.copy(), name = f'total_flux_{Spectral_window_types[i]}', dtype=float, unit=flux_unit)
        flux_err  = Column(empty_data.copy(), name = f'total_flux_e_{Spectral_window_types[i]}', dtype=float, unit=flux_err_unit)
    #   Peak_flux = MaskedColumn(empty_data.copy(), name = f'FluxPk_{Spectral_window_types[i]}', dtype=float, unit=peak_flux_unit)
        freq      = Column(empty_data.copy(), name = f'ref_freq_{Spectral_window_types[i]}', dtype=float, unit=freq_unit)
        s_code    = Column(empty_data.copy(), name = f's_code_{Spectral_window_types[i]}', dtype=str)
    #   n_gauss   = MaskedColumn(empty_data.copy(), name = f'Ngauss_{Spectral_window_types[i]}', dtype=float)
    # maybe call the global index as well
        base_window_new.add_columns([flux, flux_err, freq, s_code,])

    # will get the spw label for any source and append it to the right object 
    for index in keys_list:
        # get the values 
        spw_values = matched_dict[index]
        for spw_value in spw_values:
            spw = overlapping_windows[spw_value]['spw_id']
            base_window_new[f'total_flux_{spw}'][index] = overlapping_windows[spw_value]['total_flux']
            base_window_new[f'total_flux_e_{spw}'][index] = overlapping_windows[spw_value]['total_flux_e']
            base_window_new[f'ref_freq_{spw}'][index] = overlapping_windows[spw_value]['ref_freq']
            base_window_new[f's_code_{spw}'][index] = overlapping_windows[spw_value]['s_code']
    
    mask = np.ones(len(overlapping_windows), dtype=bool)
    mask[comparision_spw_indices] = False
    unique_sources = overlapping_windows[mask] 

    return base_window_new, unique_sources

# I will make a function that will always convert the individual indexing to a 'global' indexing'!
# just uses the dict comprehension, this allows the indexes to act as
# connecting point between steps in catalog reduction if there are intermediary steps in mutiple catalog calls for same base!

# e.g. if you have mutiple catalogs that you reduced, each will have a global index,
# for dict_catalog you would call the the reduced catalog linked to the dictionary
def global_index(dict, dict_catalog, keys_global=None):
    if keys_global != None:
        dict_global = {keys_global['Indexes'][key]: [dict_catalog['Indexes'][value] for value in value_list] 
                       for key, value_list in dict.items()}
    else:
        dict_global = {key: [dict_catalog['Indexes'][value] for value in value_list] 
                       for key, value_list in dict.items()}
    
    return dict_global
            
# this is good for only one pairing  
def ellipse(ellipse_names, ellipse_catalog):
    # hardcoded not actually good for anyting but fermi!
    mask = np.isin(ellipse_catalog['Source_Name'], ellipse_names)        
    ellipse_catalog_position = ellipse_catalog[mask]
    print(ellipse_catalog_position)
    # for each ellipse make the actual ellipse structure and use it to call 
    # this will also nullify the need to create the ellipses in 
    # defining the ellipses/circle attributes
    # the reason why it wasnt working on 6/18 is because the wrapping and negative!
    ra_ellipse = coord.Angle(ellipse_catalog_position['RAJ2000'])
    ra_ellipse = - ra_ellipse
    ra_reordered_ellipse = ra_ellipse.wrap_at(180*u.deg)
    dec_ellipse = coord.Angle(ellipse_catalog_position['DEJ2000'])

    semi_major = ellipse_catalog_position['Conf_95_SemiMajor']
    semi_minor = ellipse_catalog_position['Conf_95_SemiMinor']
    pa = ellipse_catalog_position['Conf_95_PosAng']
    
    # making all ellipses (matplotlib version) (required to actually plot the ellipses in deg in the end)
    # only way to plot it is if it is 'dimensionless' I know under the hood it is degs
    ellipse_patches = []
    for i in range(len(ellipse_catalog_position['RAJ2000'])):
            ellipses_patch = ellipse_mals((ra_reordered_ellipse[i].value, dec_ellipse[i].value),  width=semi_major[i].value*2,  height=semi_minor[i].value*2, angle=pa[i].value, lw=2, edgecolor='b', alpha=0.5, zorder=1)
            ellipse_patches.append(ellipses_patch)
    return ellipse_patches

# just from astropy tutorial 
def calc_reduced_chi_square(fit, x, y, yerr, N, n_free):
    """
    fit (array) values for the fit
    x,y,yerr (arrays) data
    N total number of points
    n_free number of parameters we are fitting
    """
    return 1.0 / (N - n_free) * sum(((fit - y) / yerr) ** 2)
