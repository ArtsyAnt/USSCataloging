# this will generate the decision tree alongside the analytics of its performance through the various hyperparameters that are looped through
# outputs a the best model as a saved file to call and reuse!
# this does not use a pca view 
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
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy.coordinates import search_around_sky
from collections import Counter

import time
import random
import os

# import seaborn as sns
import umap
from umap import UMAP

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
columns_to_keep_reduced_wise_limited  = ['Signif_Avg', 'Pivot_Energy', 'Flux1000', 'Unc_Flux1000',
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

def main(rebin_type='4'):
    # dictates which grouping to use for the rebinning process
    if rebin_type == '3':
        grouped_dict_list = grouped_dict_list_3
        rebin_label = '_3_labels'
    elif rebin_type == '4':
        grouped_dict_list = grouped_dict_list_4
        rebin_label = '_4_labels'
    print(f'Using rebinned classes: {grouped_dict_list.keys()}')

    # ========================================================= getting catalogs =======================================================
    # regroup catalogs 
    catalog_home_dir = '/lustre/aoc/observers/nm-16041/antonio/USSCataloging/catalogs/'
    catalog_reduced_dir = '/lustre/aoc/observers/nm-16041/antonio/USSCataloging/reduced_catalogs/'
    print('reading fermi catalog')
    # reading the baseline fermi sources to get the class 
    fermi_list_combined = catalog_reduced_dir  +'14_yr/' + 'MALS_spws_fermi_tables.fits'
    fermi_lat_surveys = glob.glob(catalog_home_dir +'gll**35.fit')
    survey_hud = fits.open(fermi_lat_surveys[0])
    fermi_list_14yr = QTable.read(survey_hud, format='fits')
    # group the other associated fermi sources with radio cmponents that are not the matches but in the error ellispe range (background)
    clean_class1 = list(set(fermi_list_14yr['CLASS1']))
    to_remove = ['', 'unk']
    clean_class1 = [val for val in clean_class1 if val not in to_remove]
    
    fermi_list_14yr_done = QTable.read(catalog_reduced_dir + '14yr_associated_ellipses_all_srcs_gaia_wise.fits', format='fits')
    print(len(fermi_list_14yr_done))
    fermi_list_14yr_done['CLASS1_rebinned'] = [
        next((rebin for rebin, target_cls in grouped_dict_list.items() if actual_cls in target_cls), actual_cls)
        for actual_cls in fermi_list_14yr_done['CLASS1']
    ]
    valid_values = fermi_list_14yr_done[~fermi_list_14yr_done['closest_spx_idx'].mask]
    print(f'valid sources spx idx empty reduction: {len(valid_values)}')    
    valid_values = valid_values[valid_values['closest_flux_ratio'] != np.inf]

    
    radio_alphas = []
    fermi_alphas = []
    fermi_alphas_uncertainties = []
    fermi_flux_densities = []
    fermi_flux_density_uncertanties = []
    y_lim = -3

    print(f'overlapped associated sources: {len(fermi_list_14yr_done)}')
    print(f'valid sources: {len(valid_values)}')    
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
    valid_values['wise_wise_W1_W2_color'] = valid_values['wise_wise_w1mpro'] - valid_values['wise_wise_w2mpro']
    valid_values['wise_wise_W2_W3_color'] = valid_values['wise_wise_w2mpro'] - valid_values['wise_wise_w3mpro']
    
    
    valid_values['wise_wise_W1_W4_color'] = valid_values['wise_wise_w1mpro'] - valid_values['wise_wise_w4mpro']
    valid_values['wise_wise_W2_W4_color'] = valid_values['wise_wise_w2mpro'] - valid_values['wise_wise_w4mpro']
    # 2. Cool dust component diagnostic
    valid_values['wise_wise_W3_W4_color'] = valid_values['wise_wise_w3mpro'] - valid_values['wise_wise_w4mpro']
    
    labels_to_covert =['CLASS2', 'CLASS1_rebinned', 'SpectrumType', 'CLASS1',] 

    # ============================================split data ===========================================
    print('begin preprocessing')
    '''1. preprocesing'''
    '==============================================================================='
    # remove columns in data and split between label and data
    prepreprocessed_valid_values = valid_values[columns_to_keep_reduced_wise_limited].copy()
    prepreprocessed_valid_values = prepreprocessed_valid_values.to_pandas()
    decoded_labels = {}
    # convert string class & other labels to a numerical form
    for label in labels_to_covert:
        if label == 'CLASS1':
            class1_label_encoder = LabelEncoder()
            convert = class1_label_encoder.fit_transform(prepreprocessed_valid_values[label])
            prepreprocessed_valid_values[f'converted_{label}'] = convert
        elif label == 'CLASS1_rebinned':
            class1_rebinned_label_encoder = LabelEncoder()
            convert = class1_rebinned_label_encoder.fit_transform(prepreprocessed_valid_values[label])
            prepreprocessed_valid_values[f'converted_{label}'] = convert
        
            # Create dict that links numerical labels to their class names
            class_labels = class1_rebinned_label_encoder.classes_
            class_label_values = class1_rebinned_label_encoder.transform(class_labels)
        
            for class_idx, class_str in enumerate(class_labels):
                decoded_labels[class_str] = class_label_values[class_idx]
        
            # Print the exact re-binned class -> label-index mapping
            print("\n=== REBINNED CLASS LABEL → LABEL INDEX ===")
            for idx, class_str in enumerate(class1_rebinned_label_encoder.classes_):
                print(f"Label index {idx}: {class_str}")
            print("============================================\n")
        else:
            label_coding = LabelEncoder()
            convert = label_coding.fit_transform(prepreprocessed_valid_values[label])
            prepreprocessed_valid_values[f'converted_{label}'] = convert

    # FIX: Use pandas drop method instead of .remove_columns()
    prepreprocessed_valid_values = prepreprocessed_valid_values.drop(columns=labels_to_covert)
    labels_no_group = pd.Series(prepreprocessed_valid_values['converted_CLASS1'])
    labels = pd.Series(prepreprocessed_valid_values['converted_CLASS1_rebinned'])
    # remove classes of the sources
    target_cols = ['converted_CLASS1','converted_CLASS2', 'CLASS1_rebinned', 'converted_CLASS1_rebinned']
    for target_col in target_cols:
        if target_col in prepreprocessed_valid_values.columns:
            prepreprocessed_valid_values = prepreprocessed_valid_values.drop(columns=[target_col])
    
    data_preprocessed = prepreprocessed_valid_values.replace({np.inf: np.nan, -np.inf: np.nan})
    data_preprocessed = data_preprocessed.fillna(data_preprocessed.median(numeric_only=True))
    # col_names = data_preprocessed.columns
    '================================================================================'
    scaler = StandardScaler()
    data_preprocessed_scaled = scaler.fit_transform(data_preprocessed)
    '================================================================================'
    # FULL DATA
    y_train = labels
    y_train_ungrouped = labels_no_group
    X_train = data_preprocessed_scaled 
    
    # Count occurrences of each class in your original labels
    # Identify classes that have fewer than 2 members
    class_counts = Counter(y_train_ungrouped)
    class_counts_rebinned = Counter(y_train)
    too_few_classes_ungrouped = {cls for cls, count in class_counts.items() if count < 2}
    too_few_classes_grouped = {cls for cls, count in class_counts_rebinned.items() if count < 2}
    print(f'For all CLASS1 Labels {len(too_few_classes_ungrouped)} classes has less than 2 sources and those classes are {too_few_classes_ungrouped}')
    print(f'For all regrouped class1 labels {len(too_few_classes_grouped)} classes has less than 2 sources and those classes are {too_few_classes_grouped}')

    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Step 1: Split 100% into 90% & 10% grouping
    X_remain, X_val, y_remain, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=42, stratify=y_train
    )
    # Step 2: Split the remaining 90% into 70% Train and 20% Test
    # Note: 0.2222 of 90% is exactly 20% of the original 100% (20/90 = 0.2222)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
        X_remain, y_remain, test_size=0.2222, random_state=42, stratify=y_remain
    )
    print(f'all match types: {list(decoded_labels.keys())}')
    print()
    print(f"Dataset Split Sizes -> Train: {X_train_2.shape[0]} | Test: {X_test_2.shape[0]} | Val: {X_val.shape[0]}")
    print('End preprocessing')
    print()
    
    print('Starting XGBoost Modeling')
    xgb_raw = XGBClassifier(random_state=42,
                            objective ='multi:softprob', eval_metric='mlogloss')
    print('randomized wide parameter searching')
    # large randomized search
    wide_param_dist = {
    'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400],
    'max_depth': [4,5,6,7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.7, 0.9, 1.0]
    }
    random_search = RandomizedSearchCV(
        estimator=xgb_raw,
        param_distributions=wide_param_dist,
        n_iter=100,
        scoring='f1_macro',
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train_2, y_train_2, 
    )

    # Fetch the winning parameters from Phase 1
    best_rand = random_search.best_params_
    print("Phase 1 Best:", best_rand)
    print('narrow grid parameter searching')
    
    narrow_param_grid = {
        'max_depth': sorted(list(set([max(1, best_rand['max_depth'] - 1), best_rand['max_depth'], best_rand['max_depth'] + 1]))),
        'learning_rate': sorted(list(set([max(0.005, best_rand['learning_rate'] * 0.7), best_rand['learning_rate'], min(0.5, best_rand['learning_rate'] * 1.3)]))),
        'subsample': sorted(list(set([max(0.4, best_rand['subsample'] - 0.1), best_rand['subsample'], min(1.0, best_rand['subsample'] + 0.1)]))),
        'colsample_bytree': sorted(list(set([max(0.4, best_rand['colsample_bytree'] - 0.1), best_rand['colsample_bytree'], min(1.0, best_rand['colsample_bytree'] + 0.1)]))),
        'n_estimators': sorted(list(set([max(10, int(best_rand['n_estimators'] * 0.85)), best_rand['n_estimators'], int(best_rand['n_estimators'] * 1.15)])))
    }

    grid_search = GridSearchCV(
        estimator=xgb_raw,
        param_grid=narrow_param_grid,
        scoring='f1_macro',
        cv=5,
        n_jobs=-1)
    
    grid_search.fit(X_train_2, y_train_2)
    
    print("Phase 2 Final Best:", grid_search.best_params_)
    print("Final Test Accuracy:", grid_search.best_estimator_.score(X_val, y_val))
    
        # --- PHASE 3: FINAL EVALUATION WITH EARLY STOPPING ---
    print('\nTraining final model with early stopping...')
    # Instantiate a fresh model using the absolute best parameters from Phase 2
    final_tuned_xgb = XGBClassifier(
        **grid_search.best_params_,
        early_stopping_rounds=20,  # Actually activates early stopping
        random_state=42,
        objective='multi:softprob',
        eval_metric='mlogloss'
    )
    
    # Use X_test_2 strictly as your early stopping validation set
    final_tuned_xgb.fit(
        X_train_2, y_train_2,
        eval_set=[(X_test_2, y_test_2)],
        verbose=False
    )
    # evaluate between the best fit model and the default I used before  
    # Evaluate on the true untouched holdout set (X_val)

    # compare with this simple version
    xgb_raw = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        # early_stopping_rounds=1000,
        max_depth=5,
        random_state=42,
        objective='multi:softprob',
        eval_metric='mlogloss'
    )
    xgb_raw.fit(
        X_train_2, y_train_2, 
        eval_set=[(X_test_2, y_test_2)], 
        verbose=False
    )


    xgbs = {'Best Found XGB MODEL' :final_tuned_xgb, 'Default XGB MODEL': xgb_raw}

    print("\n=== VALIDATION RESULTS SELF-TEST ===")
    for xgb_name, xgb_model in xgbs.items():
        preds_raw = xgb_model.predict_proba(X_val) # Fixed variable name from X_val_
        preds_prob_raw = xgb_model.predict_proba(X_val)
        preds_class_raw = xgb_model.predict(X_val)
        log_loss_raw = log_loss(y_val, preds_raw, labels=range(xgb_model.n_classes_))
    
        mask_actual_matches = (preds_class_raw == y_val)
        mask_actual_misses = (preds_class_raw != y_val)
        missed_probs = preds_class_raw[mask_actual_misses]
        matched_probs = preds_class_raw[mask_actual_matches]
        perc_missed = len(missed_probs)/len(y_val)
        perc_matched = len(matched_probs)/len(y_val)
        
        
        print(f"All Raw Features Validation Log Loss : {log_loss_raw:.4f}")
        print(xgb_name)
        # print(f"Optimal trees used: {xgb_model.best_iteration}")
        print("Final Test Accuracy (X_val):", xgb_model.score(X_val, y_val))
        print("Final Test Accuracy (X_test_2):", xgb_model.score(X_test_2, y_test_2))
        print(f"All Raw Features Validation Log Loss : {log_loss_raw:.4f}")
    
        print(f'Total Match Percentage {perc_matched*100}%; {len(matched_probs)}/{len(y_val)}')
        print(f'Total Missed Percentage {perc_missed*100}%; {len(missed_probs)}/{len(y_val)}')
        print()
    

        print('= Individual Class Percentages =')
        # boolean mask of the idxs of the actual individual categories 
        # loop 1. limits to appropriate indexes 2. repeat mask 3. prints out the match percentage
        for source_type in decoded_labels.keys():
            source_type_idx_mask = (y_val == decoded_labels[source_type])
            y_val_source_type = y_val[source_type_idx_mask]
            preds_class_raw_source_type = preds_class_raw[source_type_idx_mask]
            
            mask_source_type_matches = (preds_class_raw_source_type == y_val_source_type)
            matched_probs = preds_class_raw_source_type[mask_source_type_matches]
            try:
                perc_matched = len(matched_probs)/len(y_val_source_type)
                print(f'{source_type.capitalize()} Match Percentage {perc_matched*100}%; {len(matched_probs)}/{len(y_val_source_type)}')
                print()
            except:
                print(f'{source_type.capitalize()}-Class matching failed; possibly no sources in evaluation set')
                print()
                continue
        print('================================')
        # write down the associated analyitics in a file 
    print("\n=== WRITING RESULTS AND INITAL TREE MODEL ===") 
    # Save the model
    final_tuned_xgb.save_model('best_xgb_model' + rebin_label +'.json')
    print(f"Successfully saved final tuned XGBoost model to {'best_xgb_model' + rebin_label +'.json'}")

    # plt and save the feature importance
# 1. Map column names to feature importances in a Pandas Series
# (Replace 'your_column_list' with your actual text names variable)
    feat_importances = pd.Series(
        final_tuned_xgb.feature_importances_, 
        index=data_preprocessed.columns
    )
    
    # 2. Sort values so the most impactful feature sits at the top of the chart
    feat_importances_sorted = feat_importances.sort_values(ascending=True)
    
    # 3. Initialize the plot canvas with a wider layout to accommodate long names
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 4. Draw the horizontal bar chart
    feat_importances_sorted.plot(kind='barh', color='skyblue', edgecolor='black', ax=ax)
    
    # 5. Aesthetic formatting controls
    ax.set_title('XGBoost Model - Feature Importances (Gain)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Relative Importance Score', fontsize=12, labelpad=10)
    ax.set_ylabel('Features', fontsize=12, labelpad=10)
    
    # Activates background alignment lines behind the bars
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layouts to automatically fit long string arrays safely
    plt.tight_layout()
    plt.savefig('feature_importances_best_xgb_model' + rebin_label +'.png', dpi=300)
    plt.show()
    # add the second training set 
    # call it!


if __name__ == '__main__':
    main(rebin_type='4')