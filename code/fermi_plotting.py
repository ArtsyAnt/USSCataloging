from astropy.io import fits
import astropy.table as table
from astropy.table import vstack
from astropy.table import QTable, Column, MaskedColumn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import astropy.units as u
import astropy.constants as c
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

import matplotlib.colors as mcolors
import glob
import ternary


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import sklearn.cluster as cluster
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import log_loss
import warnings 

# reducing columns from the table 
columns_to_keep_reduced_wise_limited_associated  = ['Signif_Avg', 'Pivot_Energy', 'Flux1000', 'Unc_Flux1000',
                            'Energy_Flux100', 'Unc_Energy_Flux100', 'SpectrumType', 'Npred', 'Variability_Index', 'Frac_Variability', 'Unc_Frac_Variability', 'Signif_Peak', 'Flux_Peak', 'Unc_Flux_Peak', 'Time_Peak', 'Peak_Interval',
                            'CLASS1', 'CLASS2', 'closest_flux_ratio', 'closest_spx_idx', 'CLASS1_rebinned',
                            'spx_idx_ratio', 'Photon_Index', 'Photon_Index_Error', 'Fermi_Flux_Density', 'Fermi_Flux_Density_Error',

                            'wise_wise_w1mpro','wise_wise_w1sigmpro',
                            'wise_wise_w1snr','wise_wise_w1rchi2',
                            'wise_wise_w2mpro',          # W2 profile-fit magnitude
                            'wise_wise_w2sigmpro',       # W2 profile-fit magnitude uncertainty
                            'wise_wise_w2snr',           # W2 profile-fit signal-to-noise ratio
                            'wise_wise_w2rchi2',         # W2 profile-fit goodness-of-fit chi-squared
                            
                            'wise_wise_w3mpro',          # W3 profile-fit magnitude
                            'wise_wise_w3sigmpro',       # W3 profile-fit magnitude uncertainty
                            'wise_wise_w3snr',           # W3 profile-fit signal-to-noise ratio
                            'wise_wise_w3rchi2',         # W3 profile-fit goodness-of-fit chi-squared
                            
                            'wise_wise_w4mpro',          # W4 profile-fit magnitude
                            'wise_wise_w4sigmpro',       # W4 profile-fit magnitude uncertainty
                            'wise_wise_w4snr',           # W4 profile-fit signal-to-noise ratio
                            'wise_wise_w4rchi2',         # W4 profile-fit goodness-of-fit chi-squared
                            
                            'wise_wise_rchi2',           # Average goodness-of-fit chi-squared across all bands
                            'wise_wise_nb',              # Number of components fit simultaneously to the source
                            'wise_wise_na',              # Number of active pixels used in the profile fit
                            # color - color
                            'wise_wise_W1_W2_color','wise_wise_W2_W3_color', 'wise_wise_W1_W4_color','wise_wise_W2_W4_color','wise_wise_W3_W4_color']


# reducing columns from the table 
columns_to_keep_reduced_wise_limited  = ['Signif_Avg', 'Pivot_Energy', 'Flux1000', 'Unc_Flux1000',
                            'Energy_Flux100', 'Unc_Energy_Flux100', 'SpectrumType', 'Npred', 'Variability_Index', 'Frac_Variability', 'Unc_Frac_Variability', 'Signif_Peak', 'Flux_Peak', 'Unc_Flux_Peak', 'Time_Peak', 'Peak_Interval',
                            'closest_flux_ratio', 'closest_spx_idx',
                            'spx_idx_ratio', 'Photon_Index', 'Photon_Index_Error', 'Fermi_Flux_Density', 'Fermi_Flux_Density_Error',

                            'WISE_w1mpro','WISE_w1sigmpro',
                            'WISE_w1snr','WISE_w1rchi2',
                            'WISE_w2mpro',          # W2 profile-fit magnitude
                            'WISE_w2sigmpro',       # W2 profile-fit magnitude uncertainty
                            'WISE_w2snr',           # W2 profile-fit signal-to-noise ratio
                            'WISE_w2rchi2',         # W2 profile-fit goodness-of-fit chi-squared
                            
                            'WISE_w3mpro',          # W3 profile-fit magnitude
                            'WISE_w3sigmpro',       # W3 profile-fit magnitude uncertainty
                            'WISE_w3snr',           # W3 profile-fit signal-to-noise ratio
                            'WISE_w3rchi2',         # W3 profile-fit goodness-of-fit chi-squared
                            
                            'WISE_w4mpro',          # W4 profile-fit magnitude
                            'WISE_w4sigmpro',       # W4 profile-fit magnitude uncertainty
                            'WISE_w4snr',           # W4 profile-fit signal-to-noise ratio
                            'WISE_w4rchi2',         # W4 profile-fit goodness-of-fit chi-squared
                            
                            'WISE_rchi2',           # Average goodness-of-fit chi-squared across all bands
                            'WISE_nb',              # Number of components fit simultaneously to the source
                            'WISE_na',              # Number of active pixels used in the profile fit
                            # color - color
                            'WISE_W1_W2_color','WISE_W2_W3_color', 'WISE_W1_W4_color','WISE_W2_W4_color','WISE_W3_W4_color']

# overlay a bar chart of the base associated sources with the unassociated source distribution!
def bar_overlay(associated_data, top_unassociated_data, worst_unassociated_data, rebin_type,figname=None):
    """
    ----------
    associated_data : Astropy Table
        Valid associated sources.
top_unassociated_data : Astropy Table
        Top-probability unassociated matches.

    worst_unassociated_data : Astropy Table
        Worst-probability unassociated matches.

    figname: save the figure to this filename.
    """

    if rebin_type =='3':
        class_column_assoc = 'CLASS1_rebinned_3_labels'
        class_column_unassoc = 'labels_3_top_class_pred'
    else:
        class_column_assoc = 'CLASS1_rebinned_4_labels'
        class_column_unassoc = 'labels_4_top_class_pred'
    associated_classes = set(
        str(x) for x in associated_data[class_column_assoc]
        if x is not None
    )

    top_classes = set(
        str(x) for x in top_unassociated_data[class_column_unassoc]
        if x is not None)
    worst_classes = set(
        str(x) for x in worst_unassociated_data[class_column_unassoc]
        if x is not None)

    categories = sorted(
        associated_classes |
        top_classes |
        worst_classes)

    associated_counts = [
        np.sum(
            np.array(
                [str(x) for x in associated_data[class_column_assoc]]
            ) == category
        )
        for category in categories
    ]
    top_counts = [
        np.sum(
            np.array(
                [str(x) for x in top_unassociated_data[class_column_unassoc]]
            ) == category
        )
        for category in categories
    ]
    worst_counts = [
        np.sum(
            np.array(
                [str(x) for x in worst_unassociated_data[class_column_unassoc]]
            ) == category
        )
        for category in categories
    ]

    x = np.arange(len(categories))
    width = 0.25

    # plt all the bar charts for assoc and unassoc and other figure setup
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(
        x - width, associated_counts, width, color='royalblue', label='Associated')
    ax.bar(
        x, top_counts, width, color='darkorange', label='Top Match')
    ax.bar(x + width, worst_counts, width, color='crimson', label='Worst Match')


    ax.set_xlabel('Rebinned Source Class',fontsize=14)
    ax.set_ylabel('Number of Sources',fontsize=14)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45,ha='right', fontsize=12)

    ax.tick_params(axis='y',labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()

    if figname is not None:
        fig.savefig(figname, dpi=300, bbox_inches='tight')

    return fig, ax

    
def ternary_cluster(unassociated_data, figname=None):

    blazar_prob = np.asarray(unassociated_data['labels_3_blazar_prob'], dtype=float)
    other_prob = np.asarray(unassociated_data['labels_3_other_prob'], dtype=float)
    pulsar_prob = np.asarray(unassociated_data['labels_3_pulsar_prob'], dtype=float)

    valid = (
        np.isfinite(blazar_prob)
        & np.isfinite(other_prob)
        & np.isfinite(pulsar_prob)
    )

    blazar_prob = blazar_prob[valid]
    other_prob = other_prob[valid]
    pulsar_prob = pulsar_prob[valid]

    probability_sum = blazar_prob + other_prob + pulsar_prob
    points = list(zip(blazar_prob, other_prob, pulsar_prob))

    fig, tax = ternary.figure(scale=1.0, permutation="012")
    fig.set_size_inches(9, 8)

    tax.boundary(linewidth=2.0)

    tax.gridlines(
        multiple=0.2,
        color="gray",
        linestyle="--",
        linewidth=0.8
    )

    tax.scatter(
        points,
        marker="o",
        color="royalblue",
        alpha=0.5,
        s=25,
        label="Unassociated sources"
    )

    tax.left_corner_label("Blazar", fontsize=13, offset=0.10)
    tax.right_corner_label("Other", fontsize=13, offset=0.10)
    tax.top_corner_label("Pulsar", fontsize=13, offset=0.10)

    tax.left_axis_label("Blazar probability", fontsize=12, offset=0.14)
    tax.right_axis_label("Other probability", fontsize=12, offset=0.14)
    tax.bottom_axis_label("Pulsar probability", fontsize=12, offset=0.14)

    tax.ticks(axis="lbr", multiple=0.2, linewidth=1, tick_formats="%.1f")
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")

    tax.set_title(
        "XGBoost 3-Class Probabilities\nUnassociated Sources",
        fontsize=16,
        pad=20
    )

    tax.legend(loc="upper right", fontsize=11)

    tax._redraw_labels()

    if figname is not None:
        fig.savefig(figname, dpi=300, bbox_inches="tight")

    return fig, tax
    
# plot the umap cluster and show the feature importnace of all of the axi
def umap_cluster():
    fig = plt.figure(figsize=(12,12))

    ...

spectral_index_columns = [
    'RACS_low_spectral_index',
    'VLASS_spectral_index',
    'NVSS_spectral_index',
    'TGSS_spectral_index',
    'SUMSS_spectral_index',
    'FIRST_spectral_index',
    'MALS_L_spectral_index',
]

flux_ratio_columns = [
    'RACS_low_FluxRatio',
    'VLASS_FluxRatio',
    'TGSS_FluxRatio',
    'SUMSS_FluxRatio',
    'FIRST_FluxRatio',
    'MALS_L_FluxRatio',
]

# my group of four categories use the _ to differiantiate between the active and non active list
grouped_dict_list_4 = {
                     'pulsar': ['psr', 'PSR',  'msp', 'MSP'],
                     'blazar': ['bll', 'BLL',
                               'fsrq', 'FSRQ',
                               ],
                     'non_blazar_galaxy': [
                         'agn', 'AGN',
                         'css', 'CSS',
                         'nlsy1', 'NLSY1', 'rdg', 'RDG',
                         'sey', 'SEY', 'ssrq', 'SSRQ', 'gal', 
                         'GAL', 'sbg', 'SBG',
                         'bcu', 'BCU',
                     ],
                     'other' : ['snr', 'SNR',
                                'spp', 'SPP',
                                'gc', 'GC', 'glc', 'GLC',
                                'mc', 'MC', 'sfr', 'SFR',
                                'pwm', 'PWN',
                                'UNK',
                                'lmb', 'LMB', 'bin', 'BIN', 
                                'hmb', 'HMB', 'nov', 'NOV'
                               ]}
# preshanths version
grouped_dict_list_3 = {
                     'pulsar': ['psr', 'PSR',  'msp', 'MSP'],
                     'blazar': ['bll', 'BLL',
                               'fsrq', 'FSRQ',
                               ],
                     'other' : ['snr', 'SNR',
                                'spp', 'SPP',
                                'gc', 'GC', 'glc', 'GLC',
                                'mc', 'MC', 'sfr', 'SFR',
                                'pwm', 'PWN',
                                'UNK',
                                'lmb', 'LMB', 'bin', 'BIN', 
                                'hmb', 'HMB', 'nov', 'NOV',

                                 'agn', 'AGN',
                                 'css', 'CSS',
                                 'nlsy1', 'NLSY1', 'rdg',
                                 'RDG',
                                 'sey', 'SEY', 'ssrq',
                                'SSRQ', 'gal', 
                                 'GAL', 'sbg', 'SBG',
                                 'bcu', 'BCU',
                               ]}



def first_unmasked_value(row, column_names):
    for column_name in column_names:
        value = row[column_name]

        # Astropy MaskedColumn value
        if np.ma.is_masked(value):
            continue

        # Also reject NaN values if the value is unmasked but NaN
        try:
            if np.isnan(value):
                continue
        except TypeError:
            pass

        return value

    # Everything was masked/invalid
    return np.nan

# will take a fermi source or will take the stacked sources 
def main(fermi_source=None, rebin_type='4'):
    # dictates which grouping to use for the rebinning process
    # if rebin_type == '3':
    #     grouped_dict_list = grouped_dict_list_3
    #     rebin_label = '_3_labels'
    # elif rebin_type == '4':
    #     grouped_dict_list = grouped_dict_list_4
    #     rebin_label = '_4_labels'
    # print(f'Using rebinned classes: {grouped_dict_list.keys()}')
    
    catalog_base_dir = '/lustre/aoc/observers/nm-16041/antonio/USSCataloging/'
    catalog_reduced_dir = '/lustre/aoc/observers/nm-16041/antonio/USSCataloging/reduced_catalogs/'
    catalog_names = ['FERMI', 'WISE', 'RACS_low', 'TGSS', 'SUMSS', 'FIRST', 'VLASS']
    print('=== SETTING UP UNASSOCIATED & ASSOCIATED CATALOG ===')
    # setting up catalog: same process as associated counterpart
    fermi_list_14yr_done = QTable.read(catalog_reduced_dir + 'mag_prior_all_nway_matches/14/all_top_tables_stacked_probabilites.fits', format='fits')
    fermi_list_14yr_associated = QTable.read(catalog_reduced_dir + '14yr_associated_ellipses_all_srcs_gaia_wise.fits', format='fits')


    # associated - class rebinning for both 3 and 4 binning 
    print(len(fermi_list_14yr_associated))
    fermi_list_14yr_associated['CLASS1_rebinned_3_labels'] = [
        next((rebin for rebin, target_cls in grouped_dict_list_3.items() if actual_cls in target_cls), actual_cls)
        for actual_cls in fermi_list_14yr_associated['CLASS1']
    ]

    fermi_list_14yr_associated['CLASS1_rebinned_4_labels'] = [
        next((rebin for rebin, target_cls in grouped_dict_list_4.items() if actual_cls in target_cls), actual_cls)
        for actual_cls in fermi_list_14yr_associated['CLASS1']
    ]

    # target the spectral index and flux ratio columns here and recreate them for the catlaog 
    fermi_list_14yr_done['closest_spx_idx'] = [
        first_unmasked_value(row, spectral_index_columns)
        for row in fermi_list_14yr_done
    ]
    
    fermi_list_14yr_done['closest_flux_ratio'] = [
        first_unmasked_value(row, flux_ratio_columns)
        for row in fermi_list_14yr_done
    ]

    for col in fermi_list_14yr_done.colnames:
        if col.startswith('FERMI_'):
            fermi_list_14yr_done.rename_column(col, col[len('FERMI_'):])
            
    
    valid_values = fermi_list_14yr_done[
        np.isfinite(fermi_list_14yr_done['closest_spx_idx']) &
        np.isfinite(fermi_list_14yr_done['closest_flux_ratio'])]
    
    valid_values_associated =  fermi_list_14yr_associated[~fermi_list_14yr_associated['closest_spx_idx'].mask]
    valid_values_associated = valid_values_associated[valid_values_associated['closest_flux_ratio'] != np.inf]
    
    radio_alphas = []
    fermi_alphas = []
    fermi_alphas_uncertainties = []
    fermi_flux_densities = []
    fermi_flux_density_uncertanties = []
    y_lim = -3

    
    print(f'overlapped unassociated sources: {len(fermi_list_14yr_done)}')
    print(f'valid sources: {len(valid_values)}')    
    print()
    print(f'overlapped associated sources: {len(fermi_list_14yr_associated)}')
    print(f'valid sources: {len(valid_values_associated)}')    
    
    for source in valid_values:
        # Extract Fermi Gamma-Ray Index
        match source['SpectrumType']:
            case 'PowerLaw':
                fermi_alpha = float(source['PL_Index']) - 1
                fermi_alpha_uncertainty = source['Unc_PL_Index']
                fermi_flux_density = source['PL_Flux_Density']
                fermi_flux_density_uncertanty = source['Unc_PL_Flux_Density']
    
            case 'LogParabola':
                fermi_alpha = float(source['LP_Index']) - 1 
                fermi_alpha_uncertainty = source['Unc_LP_Index']
                fermi_flux_density = source['LP_Flux_Density']
                fermi_flux_density_uncertanty = source['Unc_LP_Flux_Density']
    
    
            case 'PLSuperExpCutoff':
                fermi_alpha =  float(source['PLEC_IndexS']) - 1
                fermi_alpha_uncertainty = source['Unc_PLEC_IndexS']
                fermi_flux_density = source['PLEC_Flux_Density']
                fermi_flux_density_uncertanty = source['Unc_PLEC_Flux_Density']
    
        fermi_alphas.append(fermi_alpha)
        fermi_alphas_uncertainties.append(fermi_alpha_uncertainty)
        fermi_flux_densities.append(fermi_flux_density)
        fermi_flux_density_uncertanties.append(fermi_flux_density_uncertanty)

    
    
    # redone of the fermi_alphas 
    # spectral index ratio
    valid_values['spx_idx_ratio'] = valid_values['closest_spx_idx']/valid_values['closest_spx_idx']
    valid_values['Photon_Index'] = fermi_alphas
    valid_values['Photon_Index_Error'] = fermi_alphas_uncertainties
    valid_values['Fermi_Flux_Density'] = fermi_flux_densities
    valid_values['Fermi_Flux_Density_Error'] = fermi_flux_density_uncertanties
    
    
    # wise color subtracting 
    valid_values['WISE_W1_W2_color'] = valid_values['WISE_w1mpro'] - valid_values['WISE_w2mpro']
    valid_values['WISE_W2_W3_color'] = valid_values['WISE_w2mpro'] - valid_values['WISE_w3mpro']
    
    
    valid_values['WISE_W1_W4_color'] = valid_values['WISE_w1mpro'] - valid_values['WISE_w4mpro']
    valid_values['WISE_W2_W4_color'] = valid_values['WISE_w2mpro'] - valid_values['WISE_w4mpro']
    # 2. Cool dust component diagnostic
    valid_values['WISE_W3_W4_color'] = valid_values['WISE_w3mpro'] - valid_values['WISE_w4mpro']
    
    # print current size of unassociated sources 
    print(f'valid sources: {len(valid_values)}')    

    prepreprocessed_valid_values = valid_values[columns_to_keep_reduced_wise_limited].copy()
    prepreprocessed_valid_values = prepreprocessed_valid_values.to_pandas()
    
    labels_to_covert =['SpectrumType']
    for label in labels_to_covert:
        label_coding = LabelEncoder()
        convert = label_coding.fit_transform(prepreprocessed_valid_values[label])
        prepreprocessed_valid_values[f'converted_{label}'] = convert
        
    prepreprocessed_valid_values = prepreprocessed_valid_values.drop(columns=labels_to_covert)
    # no other splitting just making is standard to get evaluated!
    data_preprocessed = prepreprocessed_valid_values.replace({np.inf: np.nan, -np.inf: np.nan})
    data_preprocessed = data_preprocessed.fillna(data_preprocessed.median(numeric_only=True))
    # col_names = data_preprocessed.columns
    '================================================================================'
    scaler = StandardScaler()
    data_preprocessed_scaled = scaler.fit_transform(data_preprocessed)

    print('=== CALLING XGBOOST MODEL ===')
    xgboost_path_3label = catalog_base_dir + 'best_xgb_model_3_labels.json'
    xgboost_path_4label = catalog_base_dir + 'best_xgb_model_4_labels.json'
    xgboost_model_3label = xgb.XGBClassifier()
    xgboost_model_3label.load_model(xgboost_path_3label)
    xgboost_model_4label = xgb.XGBClassifier()
    xgboost_model_4label.load_model(xgboost_path_4label)

    all_models = [xgboost_model_3label, xgboost_model_4label]
    print(f'xgb-classifer models loaded')
    
    prefixes = ['labels_3', 'labels_4']
    
    # Exact decoded class names corresponding to the LabelEncoder indices
    decoded_class_maps = {
        0: {
            0: 'blazar',
            1: 'other',
            2: 'pulsar',
        },
        1: {
            0: 'blazar',
            1: 'non_blazar_galaxy',
            2: 'other',
            3: 'pulsar',
        }
    }
    
    # Apply both models
    for model_idx, model in enumerate(all_models):
        prefix = prefixes[model_idx]
        # Encoded prediction: e.g. 0, 1, 2, 3
        preds_model_top_match = model.predict(data_preprocessed_scaled)
    
        # Probability for every class
        pred_model_all_probabilities = model.predict_proba(
            data_preprocessed_scaled
        )
        # Make sure predictions are integer-like
        preds_model_top_match = np.asarray(
            preds_model_top_match,
            dtype=int
        )
        # For your models this should be:
        #
        # 3-label:
        # [0, 1, 2]
        #
        # 4-label:
        # [0, 1, 2, 3]
    
        model_encoded_classes = np.asarray(
            model.classes_,
            dtype=int
        )
        print(f'\n=== {prefix} MODEL ===')
        print(f'Encoded classes: {model_encoded_classes}')
        # DECODE THE PREDICTIONS
        class_map = decoded_class_maps[model_idx]
    
        transformed_preds_model_top_match = np.array([
            class_map[int(pred)]
            for pred in preds_model_top_match
        ])
    
        # ADD TOP PREDICTION — ENCODED
        valid_values[
            prefix + '_top_class_pred_idx'
        ] = preds_model_top_match
    
        # ADD TOP PREDICTION — DECODED
        valid_values[
            prefix + '_top_class_pred'
        ] = transformed_preds_model_top_match
    
    
        for probability_column_idx, encoded_class in enumerate(
            model_encoded_classes
        ):
    
            decoded_class = class_map[int(encoded_class)]
    
            probability_column_name = (
                prefix + '_' + decoded_class + '_prob'
            )
    
            valid_values[probability_column_name] = (
                pred_model_all_probabilities[:, probability_column_idx]
            )
        print(f'{prefix} decoded classes: {list(class_map.values())}')
        
    # split the unassociated sources between the top and worst match
    top_matches_list = []
    worst_matches_list = []
    # Get unique Fermi association names
    valid_assoc = valid_values[~valid_values['ASSOC_4FGL'].mask]
    all_fermi_source_names = np.unique(valid_assoc['ASSOC_4FGL'])

    for source_name in all_fermi_source_names:
        # All matches for this Fermi source
        source_match = valid_assoc[valid_assoc['ASSOC_4FGL'] == source_name]

        # Top match
        top_match_idx = np.argmax(source_match['prob_has_match'])
        top_match = source_match[top_match_idx]
        top_matches_list.append(top_match)
    
        # Worst match
        worst_match_idx = np.argmin(source_match['prob_has_match'])
        worst_match = source_match[worst_match_idx]
        worst_matches_list.append(worst_match)   
    
    top_match_table = vstack(top_matches_list)
    worst_match_table = vstack(worst_matches_list)

    top_match_table.write(catalog_reduced_dir + 'mag_prior_all_nway_matches/14/top_stacked_matches_xgmodeled.fits', format='fits', overwrite=True)
    worst_match_table.write(catalog_reduced_dir + 'mag_prior_all_nway_matches/14/worst_stacked_matches_xgmodeled.fits', format='fits', overwrite=True)
    
    print('writing complete!')


# - will generate a barchart that overlays the unassociated source probabilites to the unassociated sources
    bar_overlay(
        associated_data=valid_values_associated,
        top_unassociated_data=top_match_table,
        worst_unassociated_data=worst_match_table,
        rebin_type ='3',
        figname=(
            catalog_reduced_dir +
            'mag_prior_all_nway_matches/14/'
            'associated_top_worst_class_label_3_distribution.png'
        )
    )

    bar_overlay(
        associated_data=valid_values_associated,
        top_unassociated_data=top_match_table,
        worst_unassociated_data=worst_match_table,
        rebin_type ='4',
        figname=(
            catalog_reduced_dir +
            'mag_prior_all_nway_matches/14/'
            'associated_top_worst_class_label_4_distribution.png'
        )
    )

# - generate a 3d clustering of the unassociated sources alongside a table of the driven features for the clustering based on umap


# generate the triangle cluster plot 
    ternary_cluster(
    valid_values,
    figname=(
        catalog_reduced_dir
        + 'mag_prior_all_nway_matches/14/'
        + 'unassociated_labels_3_ternary.png'
    )
)



 # - should be able to accept the individual fermi ellipse completley or the top three and worst match 


if __name__ =='__main__':
    main()