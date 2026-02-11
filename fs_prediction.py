# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from os.path import join
import numpy as np
import pandas as pd
import scipy.io as sio
from fs_prediction_function import fs_classifier_cv_parallel


maindir = os.path.dirname(os.path.split(os.getcwd())[0])
datadir = join(maindir,'Data/ImageDatabase_BIDS/Patients/derivatives/')
atlas = 'HCP_MMP1_DK'

df_DC = pd.read_excel(join(maindir,'Data/ImageDatabase_BIDS/masks/MRI_multicenter_MI_H.xlsx'))
df_FC = pd.read_csv(join(datadir,'TS',atlas,f'GT_all_{atlas}_prediction_mean_MI_H_atypical_ipsicontra.csv'))
df_SC = pd.read_csv(join(datadir,'MRtrix',f'GT_all_{atlas}_prediction_mean_MI_H_atypical_ipsicontra.csv'))
df_DC['Status'] = df_DC['Status'].map({'SF': 1, 'NSF': 0})
df_DC['Type of surgery'] = df_DC['Type of surgery'].map({'Res': 1,'Litt': 0})
df_DC['Surgery Side'] = df_DC['Surgery Side'].map({'L': 1,'R': 0})
df_FC['Status'] = df_FC['Status'].map({'SF': 1, 'NSF': 0})
df_FC['Type of surgery'] = df_FC['Type of surgery'].map({'Res': 1,'Litt': 0})
df_FC['Surgery Side'] = df_FC['Surgery Side'].map({'L': 1,'R': 0})
df_SC['Status'] = df_SC['Status'].map({'SF': 1, 'NSF': 0})
df_SC['Type of surgery'] = df_SC['Type of surgery'].map({'Res': 1,'Litt': 0})
df_SC['Surgery Side'] = df_SC['Surgery Side'].map({'L': 1,'R': 0})

df_FC = df_FC.rename(columns = {col: col + '_FC' for col in df_FC.columns[26:]})
df_SC = df_SC.rename(columns = {col: col + '_SC' for col in df_SC.columns[26:]})
overlap = list(set(df_FC['ID'].unique()) & set(df_SC['ID'].unique()))
df_DC = df_DC.drop(columns = ['ID','Type of surgery','Epilepsy Side','Surgery Side','Gender','Age','MTS','Scalp-EEG bilateral (temporal) spikes','iEEG bilateral (temporal) spikes','Scalp-EEG extratemporal spikes',
                              'iEEG extratemporal spikes','iEEG hemisphere','iEEG electrode type','iEEG electrode number','iEEG days of monitoring','iEEG percentage of spikes'])
df_FC = df_FC.drop(columns = ['Type of surgery','Epilepsy Side','Surgery Side','Gender','Age','MTS','Scalp-EEG bilateral (temporal) spikes','iEEG bilateral (temporal) spikes','Scalp-EEG extratemporal spikes',
                              'iEEG extratemporal spikes','iEEG hemisphere','iEEG electrode type','iEEG electrode number','iEEG days of monitoring','iEEG percentage of spikes'])
df_SC = df_SC.drop(columns = ['Type of surgery','Epilepsy Side','Surgery Side','Gender','Age','MTS','Scalp-EEG bilateral (temporal) spikes','iEEG bilateral (temporal) spikes','Scalp-EEG extratemporal spikes',
                              'iEEG extratemporal spikes','iEEG hemisphere','iEEG electrode type','iEEG electrode number','iEEG days of monitoring','iEEG percentage of spikes'])
df_FC_SC = pd.concat([df_FC[df_FC['ID'].isin(overlap)].reset_index(drop = True), 
                      df_SC[df_SC['ID'].isin(overlap)].iloc[:,11:].reset_index(drop = True)], axis = 1)
df_FC = df_FC.drop(columns = 'ID')
df_SC = df_SC.drop(columns = 'ID')
df_FC_SC = df_FC_SC.drop(columns = 'ID')
Ci = pd.read_csv(os.path.join(maindir,'Data/ImageDatabase_BIDS/masks/',atlas,f'{atlas}_Yeo7.txt'), header = None, delim_whitespace = True)
Ci = Ci.to_numpy()
Ci = np.concatenate([Ci[0:180], Ci[360:367], Ci[180:360], Ci[367:]])

subcortical_areas = ['_accumbens', '_amygdala', '_caudate', '_hippocampus', '_pallidum', '_putamen', '_thalamus']

metrics = ['PC']
outcome = 'Status' 

all_results = []
all_feature_importance = []

for i, metric in enumerate(metrics):
    for modality in ['FC-SC','FC','SC','DC']:
        if modality == 'DC' and metric == 'BC':
            df_filtered = df_DC
            df_filtered = df_filtered.drop(columns = ['GM volume','WM volume'])
        elif modality == 'FC':
            df_filtered = df_FC[df_FC['GT metrics'] == metric].reset_index(drop = True)
            df_filtered = df_filtered.drop(columns = 'GT metrics')
            brain_hubs_FC = sio.loadmat(join('/media/rgong/cMRI_BIDS/code/fs_ahubness', f"FC_{metric}_brain_hubs.mat"))[f"{metric}_final_hubs"]            
            brain_hubs_FC = (brain_hubs_FC >= 2).squeeze().astype(int)
            symmetrical_brain_hubs_FC = np.zeros_like(brain_hubs_FC)
            symmetrical_brain_hubs_FC[0:180]   = brain_hubs_FC[0:180] & brain_hubs_FC[180:360]
            symmetrical_brain_hubs_FC[180:360] = brain_hubs_FC[0:180] & brain_hubs_FC[180:360]
            symmetrical_brain_hubs_FC[360:367] = brain_hubs_FC[360:367] & brain_hubs_FC[367:]
            symmetrical_brain_hubs_FC[367:]    = brain_hubs_FC[360:367] & brain_hubs_FC[367:]
            symmetrical_brain_hubs_FC = np.concatenate([symmetrical_brain_hubs_FC[0:180], symmetrical_brain_hubs_FC[360:367], 
                                                        symmetrical_brain_hubs_FC[180:360], symmetrical_brain_hubs_FC[367:]])
            ROI_cols = df_filtered.columns[9:]
            ROI_FC_cols = [col for col in ROI_cols if col.endswith('_FC')]
            Ci_FC = np.array(Ci[symmetrical_brain_hubs_FC == 1])[:,1]
            df_filtered_FC = df_filtered.loc[:, 
                                       list(np.array(ROI_FC_cols)[np.array(symmetrical_brain_hubs_FC == 1).flatten()])].reset_index(drop = True)
            df_filtered_net_FC = {}
            for n in np.unique(Ci_FC):
                df_filtered_net_FC[f'FC_group_{n}'] = df_filtered_FC[np.array(df_filtered_FC.columns)[Ci_FC == n]].mean(axis = 1)
            df_filtered_net_FC = pd.DataFrame(df_filtered_net_FC)            
            df_filtered = pd.concat([df_filtered.iloc[:,:9].reset_index(drop = True), 
                                     df_filtered_net_FC.reset_index(drop = True)], axis = 1)
        elif modality == 'SC':
            df_filtered = df_SC[df_SC['GT metrics'] == metric].reset_index(drop = True)
            df_filtered = df_filtered.drop(columns = 'GT metrics')
            brain_hubs_SC = sio.loadmat(join('/media/rgong/cMRI_BIDS/code/fs_ahubness', f"SC_{metric}_brain_hubs.mat"))[f"{metric}_final_hubs"]            
            brain_hubs_SC = (brain_hubs_SC >= 2).squeeze().astype(int)
            symmetrical_brain_hubs_SC = np.zeros_like(brain_hubs_SC)
            symmetrical_brain_hubs_SC[0:180]   = brain_hubs_SC[0:180] & brain_hubs_SC[180:360]
            symmetrical_brain_hubs_SC[180:360] = brain_hubs_SC[0:180] & brain_hubs_SC[180:360]
            symmetrical_brain_hubs_SC[360:367] = brain_hubs_SC[360:367] & brain_hubs_SC[367:]
            symmetrical_brain_hubs_SC[367:]    = brain_hubs_SC[360:367] & brain_hubs_SC[367:]
            symmetrical_brain_hubs_SC = np.concatenate([symmetrical_brain_hubs_SC[0:180], symmetrical_brain_hubs_SC[360:367], 
                                                        symmetrical_brain_hubs_SC[180:360], symmetrical_brain_hubs_SC[367:]])
            ROI_cols = df_filtered.columns[9:]
            ROI_SC_cols = [col for col in ROI_cols if col.endswith('_SC')]
            Ci_SC = np.array(Ci[symmetrical_brain_hubs_SC == 1])[:,1]
            df_filtered_SC = df_filtered.loc[:, 
                                        list(np.array(ROI_SC_cols)[np.array(symmetrical_brain_hubs_SC == 1).flatten()])].reset_index(drop = True)
            df_filtered_net_SC = {}
            for n in np.unique(Ci_SC):
                 df_filtered_net_SC[f'SC_group_{n}'] = df_filtered_SC[np.array(df_filtered_SC.columns)[Ci_SC == n]].mean(axis = 1)
            df_filtered_net_SC = pd.DataFrame(df_filtered_net_SC)            
            df_filtered = pd.concat([df_filtered.iloc[:,:9].reset_index(drop = True), 
                                     df_filtered_net_SC.reset_index(drop = True)], axis = 1)
        elif modality == 'FC-SC':
            df_filtered = df_FC_SC[df_FC_SC['GT metrics'] == metric].reset_index(drop = True)
            df_filtered = df_filtered.drop(columns = 'GT metrics')
            brain_hubs_FC = sio.loadmat(join('/media/rgong/cMRI_BIDS/code/fs_ahubness', f"FC_{metric}_brain_hubs.mat"))[f"{metric}_final_hubs"]            
            brain_hubs_FC = (brain_hubs_FC >= 2).squeeze().astype(int)
            symmetrical_brain_hubs_FC = np.zeros_like(brain_hubs_FC)
            symmetrical_brain_hubs_FC[0:180]   = brain_hubs_FC[0:180] & brain_hubs_FC[180:360]
            symmetrical_brain_hubs_FC[180:360] = brain_hubs_FC[0:180] & brain_hubs_FC[180:360]
            symmetrical_brain_hubs_FC[360:367] = brain_hubs_FC[360:367] & brain_hubs_FC[367:]
            symmetrical_brain_hubs_FC[367:]    = brain_hubs_FC[360:367] & brain_hubs_FC[367:]
            symmetrical_brain_hubs_FC = np.concatenate([symmetrical_brain_hubs_FC[0:180], symmetrical_brain_hubs_FC[360:367], 
                                                        symmetrical_brain_hubs_FC[180:360], symmetrical_brain_hubs_FC[367:])
            ROI_cols = df_filtered.columns[9:]
            ROI_FC_cols = [col for col in ROI_cols if col.endswith('_FC')]
            Ci_FC = np.array(Ci[symmetrical_brain_hubs_FC == 1])[:,1]
            df_filtered_FC = df_filtered.loc[:, 
                                        list(np.array(ROI_FC_cols)[np.array(symmetrical_brain_hubs_FC == 1).flatten()])].reset_index(drop = True)
            df_filtered_net_FC = {}
            for n in np.unique(Ci_FC):
                 df_filtered_net_FC[f'FC_group_{n}'] = df_filtered_FC[np.array(df_filtered_FC.columns)[Ci_FC == n]].mean(axis = 1)
            df_filtered_net_FC = pd.DataFrame(df_filtered_net_FC)
            brain_hubs_SC = sio.loadmat(join('/media/rgong/cMRI_BIDS/code/fs_ahubness', f"SC_{metric}_brain_hubs.mat"))[f"{metric}_final_hubs"]            
            brain_hubs_SC = (brain_hubs_SC >= 2).squeeze().astype(int)
            symmetrical_brain_hubs_SC = np.zeros_like(brain_hubs_SC)
            symmetrical_brain_hubs_SC[0:180]   = brain_hubs_SC[0:180] & brain_hubs_SC[180:360]
            symmetrical_brain_hubs_SC[180:360] = brain_hubs_SC[0:180] & brain_hubs_SC[180:360]
            symmetrical_brain_hubs_SC[360:367] = brain_hubs_SC[360:367] & brain_hubs_SC[367:]
            symmetrical_brain_hubs_SC[367:]    = brain_hubs_SC[360:367] & brain_hubs_SC[367:]
            symmetrical_brain_hubs_SC = np.concatenate([symmetrical_brain_hubs_SC[0:180], symmetrical_brain_hubs_SC[360:367], 
                                                        symmetrical_brain_hubs_SC[180:360], symmetrical_brain_hubs_SC[367:]])
            ROI_SC_cols = [col for col in ROI_cols if col.endswith('_SC')]
            Ci_SC = np.array(Ci[symmetrical_brain_hubs_SC == 1])[:,1]
            df_filtered_SC = df_filtered.loc[:, 
                                        list(np.array(ROI_SC_cols)[np.array(symmetrical_brain_hubs_SC == 1).flatten()])].reset_index(drop = True)
            df_filtered_net_SC = {}
            for n in np.unique(Ci_SC):
                 df_filtered_net_SC[f'SC_group_{n}'] = df_filtered_SC[np.array(df_filtered_SC.columns)[Ci_SC == n]].mean(axis = 1)
            df_filtered_net_SC = pd.DataFrame(df_filtered_net_SC)            
            df_filtered = pd.concat([df_filtered.iloc[:,:9].reset_index(drop = True), 
                                     df_filtered_net_FC.reset_index(drop = True), 
                                     df_filtered_net_SC.reset_index(drop = True)], axis = 1)
                                         
        test_results, shap_values_sets, odds_ratio_sets, auc_ci_sets, npv_ci_sets, features = fs_classifier_cv_parallel(df_filtered, metric, modality)
        average_shap_importance = np.mean([np.abs(shap_values).mean(axis = 0) 
                                           for shap_values in shap_values_sets], 
                                           axis = 0)
        std_shap_importance = np.std([np.abs(shap_values).mean(axis = 0) 
                                      for shap_values in shap_values_sets], 
                                      axis = 0)
        average_odds_ratio = np.array([odds_ratio.flatten() for odds_ratio in odds_ratio_sets]).mean(axis = 0) 
        std_odds_ratio = np.array([odds_ratio.flatten() for odds_ratio in odds_ratio_sets]).std(axis = 0) 
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'shap': average_shap_importance,
            'shap sd': std_shap_importance,
            'odds ratio': average_odds_ratio,
            'odds ratio sd': std_odds_ratio
             })
        feature_importance_df = feature_importance_df.sort_values(by = 'shap', 
                                                                  ascending = False)
        feature_importance_df['modality'] = modality
        feature_importance_df['metric'] = metric
        auc_ci_sets = pd.DataFrame(auc_ci_sets)
        npv_ci_sets = pd.DataFrame(npv_ci_sets)
        test_results.to_csv(join(maindir,'Results','fs_prediction','tables',f'{metric}_{modality}_wo_FC6_HCP_MMP1_DK_all.csv'))
        feature_importance_df.to_csv(join(maindir,'Results','fs_prediction','tables',f'FI_{metric}_{modality}_HCP_MMP1_DK_all.csv'))
        auc_ci_sets.to_csv(join(maindir,'Results','fs_prediction','tables',f'auc_ci_{metric}_{modality}_HCP_MMP1_DK_all.csv'))
        npv_ci_sets.to_csv(join(maindir,'Results','fs_prediction','tables',f'npv_ci_{metric}_{modality}_HCP_MMP1_DK_all.csv'))

        
