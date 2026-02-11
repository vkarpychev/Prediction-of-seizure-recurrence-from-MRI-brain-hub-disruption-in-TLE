# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from os.path import join
import fnmatch
import pandas as pd
import numpy as np
from nilearn.image import load_img 
from nilearn.glm import regression
import nibabel as nb


maindir = os.path.dirname(os.path.split(os.getcwd())[0])
datadir = join(maindir,'Data/ImageDatabase_BIDS/Patients/derivatives/fmriprep')
patient_list = sorted([x for x in fnmatch.filter(os.listdir(datadir),'sub-*') if '.html' not in x])
ses = 'ses-pre'

for patient in patient_list:

#    file_task = sorted(fnmatch.filter(os.listdir(join(datadir,patient,ses,'func')),'*WordGeneration*_timeseries.tsv'))
    
#    fmri_task_path = sorted(fnmatch.filter(os.listdir(join(datadir,patient,ses,'func')),'*WordGeneration*_desc-preproc_bold.nii.gz'))
    
#    mask_path = sorted(fnmatch.filter(os.listdir(join(datadir,patient,ses,'func')),'*WordGeneration*_res-2_desc-brain_mask.nii.gz'))
            
#    fmri_img = load_img(join(datadir,patient,ses,'func', str(fmri_task_path)[2:-2]))

#    confound = fnmatch.filter(os.listdir(join(datadir,patient,ses,'func')),file_task[0])

#    confound = pd.read_csv(join(datadir,patient,ses,'func', str(confound)[2:-2]), sep = '\t')
    
#    for col in confound.columns:
#        if sum(confound[col].isnull()) > 0:
#            confound[col] = confound[col].fillna(np.mean(confound[col]))
            
#    model = [['global_signal'] + confound.filter(like = 'trans_').columns.tolist() + confound.filter(like = 'rot_').columns.tolist() + 
#                    confound.filter(like = 'cosine').columns.tolist() + confound.filter(like = 'a_comp_cor_').columns[:10].tolist()]
    
#    confound_pipeline = confound[model[0]]
        
#    confound_pipeline['constant'] = 1
    
#    data = np.reshape(fmri_img.get_fdata(), (-1, confound_pipeline.values.shape[0]))
            
#    model = regression.OLSModel(confound_pipeline.values) 
    
#    results = model.fit(data.T)
    
#    data_cleaned = results.residuals.T + np.reshape(np.mean(data, axis = 1), (len(np.mean(data, axis = 1)), 1))
        
#    fmri_img_cleaned = np.reshape(data_cleaned, fmri_img.shape).astype('float32')
    
#    fmri_img_cleaned = nb.Nifti1Image(fmri_img_cleaned, fmri_img.affine, header = fmri_img.header)
    
#    fmri_img_cleaned.to_filename(join(datadir,patient,'%s/func/%s_%s_task-WordGeneration_space-MNI152NLin6Asym_res-2_desc-preproc-clean_bold.nii' %(ses,patient,ses)))
        
#    os.system(str('3dTstat -mask %s -prefix %s %s -overwrite' 
#                      % (join(datadir,patient,ses,'func', str(mask_path)[2:-2]),
#                         join(datadir,patient,ses,'func', str(fmri_task_path)[2:-14] + '-mean_bold.nii'),
#                         join(datadir,patient,ses,'func', str(fmri_task_path)[2:-14] + '-clean_bold.nii'))))    
    
#    os.system(str('3dcalc -a %s -b %s -c %s -exp "c * min(200, a/b*100)*step(a)*step(b)" -prefix %s -overwrite'
#                  % (join(datadir,patient,ses,'func', str(fmri_task_path)[2:-14] + '-clean_bold.nii'), 
#                  join(datadir,patient,ses,'func', str(fmri_task_path)[2:-14] + '-mean_bold.nii'), 
#                  join(datadir,patient,ses,'func', str(mask_path)[2:-2]),
#                  join(datadir,patient,ses,'func', str(fmri_task_path)[2:-14] + '-scale_bold.nii'))))
        
#    os.system(str('3dBlurToFWHM -FWHM 8 -mask %s -prefix %s -input %s -overwrite'
#                  % (join(datadir,patient,ses,'func', str(mask_path)[2:-2]),
#                  join(datadir,patient,ses,'func', str(fmri_task_path)[2:-14] + '-smooth_bold.nii'),
#                  join(datadir,patient,ses,'func', str(fmri_task_path)[2:-14] + '-clean_bold.nii'))))
    
#    del file_task, fmri_task_path, mask_path
    
    file_rest = fnmatch.filter(os.listdir(join(datadir,patient,ses,'func')),'*rest*_timeseries.tsv')

    fmri_rest_path = fnmatch.filter(os.listdir(join(datadir,patient,ses,'func')),'*rest*_res-2_desc-preproc_bold.nii.gz')

    mask_path = fnmatch.filter(os.listdir(join(datadir,patient,ses,'func')),'*rest*_res-2_desc-brain_mask.nii.gz')      
    
    fmri_img = load_img(join(datadir,patient,ses,'func', str(fmri_rest_path)[2:-2]))

    confound = fnmatch.filter(os.listdir(join(datadir,patient,ses,'func')), file_rest[0])

    confound = pd.read_csv(join(datadir,patient,ses,'func', str(confound)[2:-2]), sep = '\t')
    
    for col in confound.columns:
    
        if sum(confound[col].isnull()) > 0:
                    
            confound[col] = confound[col].fillna(np.mean(confound[col]))
        
        model = [confound.filter(like = 'trans_').columns.tolist() + confound.filter(like = 'rot_').columns.tolist() +          
                 confound.filter(like = 'csf').columns.tolist() + confound.filter(like = 'white_matter').columns.tolist() +                 
                 confound.filter(like = 'global_signal').columns.tolist() + confound.filter(like = 'cosine').columns.tolist()]
                 
    model[0].remove('csf_wm')
        
    confound_pipeline = confound[model[0]]
    
    confound_pipeline['constant'] = 1
    
    data = np.reshape(fmri_img.get_fdata(), (-1, confound_pipeline.values.shape[0]))
        
    model = regression.OLSModel(confound_pipeline.values) 
 
    results = model.fit(data.T)
 
    data_cleaned = results.residuals.T + np.reshape(np.mean(data, axis = 1), (len(np.mean(data, axis = 1)), 1))
 
    fmri_img_cleaned = np.reshape(data_cleaned, fmri_img.shape).astype('float32')
 
    fmri_img_cleaned = nb.Nifti1Image(fmri_img_cleaned, fmri_img.affine, header = fmri_img.header)
 
    fmri_img_cleaned.to_filename(join(datadir,patient,'%s/func/%s_%s_task-rest_acq-APdespike_space-MNI152NLin6Asym_res-2_desc-preproc-clean_bold.nii.gz' 
                                      %(ses,patient,ses)))
 
    os.system(str('3dTstat -mask %s -prefix %s %s -overwrite' 
                  % (join(datadir,patient,ses,'func', str(mask_path)[2:-2]),
                     join(datadir,patient,ses,'func', str(fmri_rest_path)[2:-14] + '-mean_bold.nii.gz'),
                     join(datadir,patient,ses,'func', str(fmri_rest_path)[2:-14] + '-clean_bold.nii.gz'))))    
 
    os.system(str('3dcalc -a %s -b %s -c %s -exp "c * min(200, a/b*100)*step(a)*step(b)" -prefix %s -overwrite'
              % (join(datadir,patient,ses,'func', str(fmri_rest_path)[2:-14] + '-clean_bold.nii.gz'), 
              join(datadir,patient,ses,'func', str(fmri_rest_path)[2:-14] + '-mean_bold.nii.gz'), 
              join(datadir,patient,ses,'func', str(mask_path)[2:-2]),
              join(datadir,patient,ses,'func', str(fmri_rest_path)[2:-14] + '-scale_bold.nii.gz'))))
    
    os.system(str('3dBlurToFWHM -FWHM 8 -mask %s -prefix %s -input %s -overwrite'
              % (join(datadir,patient,ses,'func', str(mask_path)[2:-2]),
              join(datadir,patient,ses,'func', str(fmri_rest_path)[2:-14] + '-smooth_bold.nii.gz'),
              join(datadir,patient,ses,'func', str(fmri_rest_path)[2:-14] + '-clean_bold.nii.gz'))))
 
    del file_rest, fmri_rest_path, mask_path

    