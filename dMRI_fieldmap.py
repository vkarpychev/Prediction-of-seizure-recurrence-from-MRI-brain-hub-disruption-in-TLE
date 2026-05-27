#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:04:30 2023

@author: victor
"""

import os
from os.path import join
import fnmatch
import numpy as np
import nibabel as nib
import json


maindir = os.path.dirname(os.path.split(os.getcwd())[0])
datadir = join(maindir,'MRI_BIDS/Data/ImageDatabase_BIDS/Patients/') # the payway to the data
patient_list = sorted([x for x in fnmatch.filter(os.listdir(datadir),'sub-*')])
ses = 'ses-pre'
os.system(str('echo "zekelab_ieeg" | sudo -S usermod -aG docker rgong'))
os.system(str('newgrp docker'))

for patient in patient_list:

    dmri_path = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'dwi')),'*_dwi*.nii.gz'))
    anat_path = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'anat')),'*_T1w.nii.gz'))
    json_path = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'dwi')),'*_dwi*.json'))
    bval_paths = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'dwi')),'*_dwi*.bval'))
    bvec_paths = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'dwi')),'*_dwi*.bvec'))
    
    if len(dmri_path) > 1:
        os.system(str('fslmerge -t %s %s %s' %(join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')),
                                           join(datadir,patient,ses,'dwi', str(dmri_path[0])),   
                                           join(datadir,patient,ses,'dwi', str(dmri_path[1])))))
    else:
        os.system(str('cp %s %s' %(join(datadir,patient,ses,'dwi', str(dmri_path[0])), 
                                   join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')))))
    
    os.system(str('cp %s %s' %(join(datadir,patient,ses,'dwi', str(json_path[0])), 
                               join(datadir,patient,ses,'dwi', str(json_path[0]).replace('_dir-AP','')))))
    
    dmri = nib.load(join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')))
    if dmri.shape[2] % 2 != 0:
        os.system(str('fslroi %s %s 0 -1 0 -1 1 -1' %(join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')),
                                                      join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')))))
    for i, bval_path in enumerate(bval_paths):
        if i == 0:
            bval = np.loadtxt(join(datadir,patient,ses,'dwi',bval_paths[i]))
            bvec = np.loadtxt(join(datadir,patient,ses,'dwi',bvec_paths[i]))
        else:
            bval = np.concatenate((bval, np.loadtxt(join(datadir,patient,ses,'dwi',bval_paths[i]))))
            bvec = np.concatenate((bvec, np.loadtxt(join(datadir,patient,ses,'dwi',bvec_paths[i]))), axis = 1)
            
    np.savetxt(join(datadir,patient,ses,'dwi', str(bval_paths[0]).replace('_dir-AP','')[0:-5] + '_bval.txt'), bval.reshape(1,-1), fmt = '%d')
    np.savetxt(join(datadir,patient,ses,'dwi', str(bvec_paths[0]).replace('_dir-AP','')[0:-5] + '_bvec.txt'), bvec, fmt = '%.5f')    
    b0s = np.where(bval <= 5)[0]
    
    for j, b0 in enumerate(b0s):
        if j == 0:
            os.system(str('fslroi %s %s %s 1' %(join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')),
                                                join(datadir,patient,ses,'dwi', 'b0s_' + str(dmri_path[0]).replace('_dir-AP','')), str(b0))))
        else:
            os.system(str('fslroi %s %s %s 1' %(join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')),
                                                join(datadir,patient,ses,'dwi', 'b0_' + str(dmri_path[0]).replace('_dir-AP','')), str(b0))))
            
            os.system(str('fslmerge -t %s %s %s' %(join(datadir,patient,ses,'dwi', 'b0s_' + str(dmri_path[0]).replace('_dir-AP','')),
                                                   join(datadir,patient,ses,'dwi', 'b0s_' + str(dmri_path[0]).replace('_dir-AP','')),
                                                   join(datadir,patient,ses,'dwi', 'b0_' + str(dmri_path[0]).replace('_dir-AP','')))))
            
    os.system(str('fslmaths %s -Tmean %s' %(join(datadir,patient,ses,'dwi', 'b0s_' + str(dmri_path[0]).replace('_dir-AP','')),
                                            join(datadir,patient,ses,'dwi', 'b0_mean_' + str(dmri_path[0]).replace('_dir-AP','')))))
    os.system(str('mkdir %s' %(join(datadir,patient,ses,'dwi','input'))))
    os.system(str('mkdir %s' %(join(datadir,patient,ses,'dwi','output'))))
    os.system(str('cp %s %s' %(join(datadir,patient,ses,'anat',str(anat_path)[2:-2]), 
                               join(datadir,patient,ses,'dwi','input','T1.nii.gz'))))
    os.system(str('cp %s %s' %(join(datadir,patient,ses,'dwi', 'b0_mean_' + str(dmri_path[0]).replace('_dir-AP','')), 
                               join(datadir,patient,ses,'dwi','input','b0.nii.gz'))))
    f = open(join(datadir,patient,ses,'dwi',str(json_path[0]).replace('_dir-AP','')))
    data = json.load(f)
    dwell_time = (data['PhaseEncodingSteps'] - 1) * data['EffectiveEchoSpacing']
    
    acqparams = f"0 1 0 {dwell_time}\n0 1 0 0.000\n"
    with open(join(datadir,patient,ses,'dwi','input','acqparams.txt'), 'w') as file:
        file.write(acqparams)
        
    num_volumes = dmri.shape[3]
    with open(join(datadir,patient,ses,'dwi','input','index.txt'), 'w') as file:
        file.write('1 ' * num_volumes)
            
    os.system(str('echo "zekelab_ieeg" | sudo -S docker run --rm -v %s -v %s -v /usr/local/freesurfer/7.3.2/license.txt:/extra/freesurfer/license.txt' 
                  ' leonyichencai/synb0-disco:v3.1 /INPUTS /OUTPUTS' 
                  %(join(datadir,patient,ses,'dwi','input:','INPUTS'),
                    join(datadir,patient,ses,'dwi','output:','OUTPUTS'))))
    os.system(str('fslroi %s %s 1 1' %(join(datadir,patient,ses,'dwi','output','b0_all_topup.nii.gz'),
                                       join(datadir,patient,ses,'dwi','input','b0_topup.nii.gz'))))
    os.system(str('bet %s %s -f 0.35' %(join(datadir,patient,ses,'dwi','input','b0_topup.nii.gz'),
                                join(datadir,patient,ses,'dwi','input','b0_mask.nii.gz'))))
    os.system(str('fslmaths %s -bin %s' %(join(datadir,patient,ses,'dwi','input','b0_mask.nii.gz'),
                                          join(datadir,patient,ses,'dwi','input','b0_mask_bin.nii.gz'))))
    os.system(str('eddy --data_is_shelled --imain=%s --mask=%s --acqp=%s --index=%s --bvecs=%s --bvals=%s --topup=%s --out=%s' 
                  %(join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')),
                    join(datadir,patient,ses,'dwi','input','b0_mask_bin.nii.gz'),
                    join(datadir,patient,ses,'dwi','input','acqparams.txt'),
                    join(datadir,patient,ses,'dwi','input','index.txt'),
                    join(datadir,patient,ses,'dwi', str(bvec_paths[0]).replace('_dir-AP','')[0:-5] + '_bvec.txt'),  
                    join(datadir,patient,ses,'dwi', str(bval_paths[0]).replace('_dir-AP','')[0:-5] + '_bval.txt'),
                    join(datadir,patient,ses,'dwi','output','topup'),
                    join(datadir,patient,ses,'dwi', 'output', str(dmri_path[0]).replace('_dir-AP','')[0:-10] + 'dir-APeddy_dwi.nii.gz'))))
    os.system('cp %s %s' %(join(datadir,patient,ses,'dwi', 'output', str(dmri_path[0]).replace('_dir-AP','')[0:-10] + 'dir-APeddy_dwi.nii.gz'),
                           join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')[0:-10] + 'dir-APeddy_dwi.nii.gz')))
    os.system('cp %s %s' %(join(datadir,patient,ses,'dwi', 'output', str(dmri_path[0]).replace('_dir-AP','')[0:-10] + 'dir-APeddy_dwi.nii.gz.eddy_rotated_bvecs'),
                           join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')[0:-10] + 'dir-APeddy_dwi_bvec.txt')))
    os.system('cp %s %s' %(join(datadir,patient,ses,'dwi', str(bvec_paths[0]).replace('_dir-AP','')[0:-5] + '_bval.txt'),
                           join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')[0:-10] + 'dir-APeddy_dwi_bval.txt')))
    os.system('cp %s %s' %(join(datadir,patient,ses,'dwi', str(json_path[0]).replace('_dir-AP','')),
                           join(datadir,patient,ses,'dwi', str(dmri_path[0]).replace('_dir-AP','')[0:-10] + 'dir-APeddy_dwi.json')))
    os.system('cp %s %s' %(join(datadir,patient,ses,'dwi','input','b0_mask_bin.nii.gz'),
                           join(datadir,patient,ses,'dwi','b0_mask_bin.nii.gz')))
    os.system(str('rm -r %s %s' %(join(datadir,patient,ses,'dwi','input'),
                                      join(datadir,patient,ses,'dwi','output'))))
    os.system(str('rm -r %s %s %s' %(join(datadir,patient,ses,'dwi', 'b0s_' + str(dmri_path[0]).replace('_dir-AP','')),
                                           join(datadir,patient,ses,'dwi', 'b0_' + str(dmri_path[0]).replace('_dir-AP','')),
                                           join(datadir,patient,ses,'dwi', 'b0_mean_' + str(dmri_path[0]).replace('_dir-AP','')))))
    
    del dmri, num_volumes, i, j, f, acqparams, dwell_time, bvec, bval, b0s, b0
