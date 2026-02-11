#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:04:30 2023

@author: victor
"""

import os
from os.path import join
import fnmatch
from nipype.interfaces import afni
import json
import re


maindir = os.path.dirname(os.path.split(os.getcwd())[0])

datadir = join(maindir,'MRI_BIDS/Data/ImageDatabase_BIDS/Patients/')

patient_list = sorted([x for x in fnmatch.filter(os.listdir(datadir),'sub-*')])
ses = 'ses-pre'
os.system(str('echo "zekelab_ieeg" | sudo -S usermod -aG docker rgong'))
os.system(str('newgrp docker'))

for patient in patient_list:

    fmri_path = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'func')),'*_acq-AP_bold.nii.gz'))
    
    anat_path = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'anat')),'*_T2w.nii.gz'))
    
    json_path = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'func')),'*_acq-AP_bold.json'))
    
    for run, path in enumerate(fmri_path):
        
        f = open(join(datadir,patient,ses,'func',str(json_path[run])))
        
        data_read = json.load(f)
        
        tshift = afni.TShift()
        
        tshift.inputs.in_file = join(datadir,patient,ses,'func',str(fmri_path[run]))
                        
        tshift.inputs.slice_timing = data_read['SliceTiming']
        
        TR = data_read['RepetitionTime']
                        
        tshift.cmdline
        
        match_despike1 = re.split(r"(_bold)", str(fmri_path[run]))
        match_despike2 = re.split(r"(_bold)", str(json_path[run]))
        
        os.system(str('3dDespike -overwrite -prefix %s %s' 
                      %(join(datadir,patient,ses,'func',(match_despike1[0] + 'despike' + match_despike1[1] + match_despike1[2])),
                        join(datadir,patient,ses,'func',str(fmri_path[run])))))
        
        os.system(str('cp %s %s' %(join(datadir,patient,ses,'func',str(json_path)[2:-2]),
                                  (join(datadir,patient,ses,'func', (match_despike2[0] + 'despike' + match_despike2[1] + match_despike2[2]))))))
        
        os.system(str('3dTshift -overwrite -quintic -TR ' + str(TR) + ' -tpattern @slice_timing.1D -prefix %s %s'
                  % (join(datadir,patient,ses,'func',(match_despike1[0] + 'despike' + match_despike1[1] + match_despike1[2])),
                     join(datadir,patient,ses,'func',(match_despike1[0] + 'despike' + match_despike1[1] + match_despike1[2])))))
        
        os.system(str('mkdir %s' %(join(datadir,patient,ses,'func','input'))))
        os.system(str('mkdir %s' %(join(datadir,patient,ses,'func','output'))))
        
        os.system(str('cp %s %s' %(join(datadir,patient,ses,'func',(match_despike1[0] + 'despike' + match_despike1[1] + match_despike1[2])), 
                                   join(datadir,patient,ses,'func','input','BOLD_d.nii.gz'))))
        
        os.system(str('cp %s %s' %(join(datadir,patient,ses,'anat',str(anat_path)[2:-2]), 
                                   join(datadir,patient,ses,'func','input','T1.nii.gz'))))
                
        os.system(str('echo "zekelab_ieeg" | sudo -S docker run --rm -v %s -v %s -v /usr/local/freesurfer/7.3.2/license.txt:/opt/freesurfer/license.txt' 
                     ' -v /tmp:/tmp ytzero/synbold-disco:v1.4 /INPUTS /OUTPUTS --motion_corrected -no_topup --no_smoothing' 
                     %(join(datadir,patient,ses,'func','input:','INPUTS'),
                       join(datadir,patient,ses,'func','output:','OUTPUTS'))))
                
        os.system(str('mkdir %s' %(join(datadir,patient,ses,'fmap'))))
        
        os.system(str('cp %s %s' %(join(datadir,patient,ses,'func','output','BOLD_s_3D.nii.gz'),
                                   join(datadir,patient,ses,'fmap', str(patient) + '_ses-pre_acq-synbold_dir-PA_epi.nii.gz'))))
         
        data_write = {'TotalReadoutTime': 1.0,
                      'PhaseEncodingDirection': 'j',
                      'IntendedFor': str('%s' %(join(ses,"func",(match_despike1[0] + "despike" + match_despike1[1] + match_despike1[2]))))}
        
        with open(join(datadir,patient,ses,'fmap', str(patient) + '_ses-pre_acq-synbold_dir-PA_epi.nii.gz')[0:-7] + str('.json'), 'w') as json_file:
            json.dump(data_write, json_file, indent = 4) 
        
        os.system(str('rm -r %s %s' %(join(datadir,patient,ses,'func','input'),
                                      join(datadir,patient,ses,'func','output'))))
                
        del f, data_read, data_write, tshift, TR, match_despike1, match_despike2
        
        
