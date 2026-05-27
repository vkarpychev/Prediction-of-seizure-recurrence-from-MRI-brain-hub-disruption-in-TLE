#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 15:29:45 2024

@author: lambda
"""

import os
from os.path import join
import fnmatch
import numpy as np
import subprocess
import re
import shutil


def dMRI_MRtrix_patient(patient):
    
    print(f'Processing {patient} ')
    maindir = os.path.dirname(os.path.split(os.getcwd())[0])
    datadir = join(maindir,'Data/ImageDatabase_BIDS/Patients/') 
    resultdir = join(maindir,'Data/ImageDatabase_BIDS/Patients/derivatives/MRtrix')
    ses = 'ses-pre'
    
    dmri_path = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'dwi')),'*-APeddy_dwi.nii.gz'))
    bval_path = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'dwi')),'*-APeddy_dwi_bval.txt'))
    bvec_path = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'dwi')),'*-APeddy_dwi_bvec.txt'))
    anat_path = sorted(fnmatch.filter(os.listdir(join(datadir, patient, ses,'anat')),'*_T1w.nii.gz'))
 
    # if os.path.exists(join(resultdir,patient)) == 0: 
    #     os.system(str('mkdir %s' %(join(resultdir,patient))))
    # os.system(str('cp %s %s' %(join(datadir, patient, ses,'dwi', str(dmri_path)[2:-2]),
    #                            join(resultdir,patient,'dwi.nii.gz'))))
    # os.system(str('cp %s %s' %(join(datadir, patient, ses,'dwi', str(bval_path)[2:-2]),
    #                            join(resultdir,patient,'dwi.bval'))))
    # os.system(str('cp %s %s' %(join(datadir, patient, ses,'dwi', str(bvec_path)[2:-2]),
    #                           join(resultdir, patient,'dwi.bvec'))))
    # os.system(str('cp %s %s' %(join(datadir, patient, ses,'anat', str(anat_path)[2:-2]),
    #                            join(resultdir,patient,'T1.nii.gz'))))
    # os.system(str('mrconvert %s %s -fslgrad %s %s -nthreads 24 -force' %(join(resultdir,patient,'dwi.nii.gz'),
    #                                                                      join(resultdir,patient,'dwi.mif'),
    #                                                                      join(resultdir,patient,'dwi.bvec'),
    #                                                                      join(resultdir,patient,'dwi.bval'))))
    # os.system(str('dwiextract %s - -bzero | mrmath - mean %s -axis 3 -nthreads 24 -force' %(join(resultdir,patient,'dwi.mif'),
    #                                                                                         join(resultdir,patient,'mean_b0.nii.gz'))))
    # os.system(str('python /home/rgong/synthstrip-docker -i %s -m  %s' %(join(resultdir,patient,'mean_b0.nii.gz'),
    #                                                                     join(resultdir,patient,'brain_mask.nii.gz'))))
    # os.system(str('mrconvert %s %s -nthreads 24 -force' %(join(resultdir,patient,'brain_mask.nii.gz'),
    #                                                       join(resultdir,patient,'mask.mif'))))
    # os.system(str('mrconvert %s %s -nthreads 24 -force' %(join(resultdir,patient,'mean_b0.nii.gz'),
    #                                                       join(resultdir,patient,'mean_b0.mif'))))
    # status = os.system(f"echo 'zekelab_ieeg' | docker run --rm -v {resultdir}:/data mrtrix3/mrtrix3 "
    #                    f"dwibiascorrect ants /data/{patient}/dwi.mif /data/{patient}/dwi_unbiased_tmp.mif "
    #                    f"-bias /data/{patient}/bias.mif -mask /data/{patient}/mask.mif "
    #                    f"-nthreads 24 -force")
    # if status != 0:
    #     shutil.copyfile(
    #         join(resultdir, patient, 'dwi.mif'),
    #         join(resultdir, patient, 'dwi_unbiased_tmp.mif')
    #         )
    # os.system(str('mrconvert %s %s -fslgrad %s %s -nthreads 24 -force' %(
    #      join(resultdir, patient, 'dwi_unbiased_tmp.mif'),
    #      join(resultdir, patient, 'dwi_unbiased.mif'),
    #      join(resultdir, patient, 'dwi.bvec'),
    #      join(resultdir, patient, 'dwi.bval'))))
    # os.system('rm -r %s' %(join(resultdir, patient, 'dwi_unbiased_tmp.mif')))
    # bval = np.loadtxt(join(resultdir,patient,'dwi.bval'))
    # unique_bval, counts = np.unique(np.round(bval, -2), return_counts = True)
    # unique_bval = unique_bval[unique_bval > 5]
    # if len(unique_bval) < 2:
    #     os.system(str('dwi2response tournier %s %s -voxels %s -mask %s -nthreads 24 -force' %(join(resultdir,patient,'dwi_unbiased.mif'),
    #                                                                                           join(resultdir,patient,'wm.txt'),
    #                                                                                           join(resultdir,patient,'voxels.mif'),
    #                                                                                           join(resultdir,patient,'mask.mif'))))
    #     os.system(str('dwi2fod csd %s %s %s -mask %s -nthreads 24 -force' %(join(resultdir,patient,'dwi_unbiased.mif'),
    #                                                                         join(resultdir,patient,'wm.txt'),
    #                                                                         join(resultdir,patient,'wmfod.mif'),
    #                                                                         join(resultdir,patient,'mask.mif'))))
    #     os.system(str('mtnormalise %s %s -mask %s -info -nthreads 24 -force' %(join(resultdir,patient,'wmfod.mif'),
    #                                                                            join(resultdir,patient,'wmfod_norm.mif'),
    #                                                                            join(resultdir,patient,'mask.mif')))) 
    # else:
    #     os.system(str('dwi2response dhollander %s %s %s %s -voxels %s -mask %s -nthreads 24 -force' %(join(resultdir,patient,'dwi_unbiased.mif'),
    #                                                                                                   join(resultdir,patient,'wm.txt'),
    #                                                                                                   join(resultdir,patient,'gm.txt'),
    #                                                                                                   join(resultdir,patient,'csf.txt'),
    #                                                                                                   join(resultdir,patient,'voxels.mif'),
    #                                                                                                   join(resultdir,patient,'mask.mif'))))
    #     os.system(str('dwi2fod msmt_csd %s -mask %s %s %s %s %s %s %s -nthreads 24 -force' %(join(resultdir,patient,'dwi_unbiased.mif'),
    #                                                                                         join(resultdir,patient,'mask.mif'),
    #                                                                                         join(resultdir,patient,'wm.txt'),                                                                                      
    #                                                                                         join(resultdir,patient,'wmfod.mif'),
    #                                                                                         join(resultdir,patient,'gm.txt'),
    #                                                                                         join(resultdir,patient,'gmfod.mif'),
    #                                                                                         join(resultdir,patient,'csf.txt'),
    #                                                                                         join(resultdir,patient,'csffod.mif'))))
    #     os.system(str('mtnormalise %s %s %s %s %s %s -mask %s -nthreads 24 -force' %(join(resultdir,patient,'wmfod.mif'),
    #                                                                                 join(resultdir,patient,'wmfod_norm.mif'),
    #                                                                                 join(resultdir,patient,'gmfod.mif'),                                                                              
    #                                                                                 join(resultdir,patient,'gmfod_norm.mif'),
    #                                                                                 join(resultdir,patient,'csffod.mif'),
    #                                                                                 join(resultdir,patient,'csffod_norm.mif'),
    #                                                                                 join(resultdir,patient,'mask.mif'))))
    # os.system(str('mrconvert %s %s -nthreads 24 -force' %(join(resultdir,patient,'T1.nii.gz'),
    #                                                       join(resultdir,patient,'T1.mif'))))
    # os.system(str('5ttgen fsl %s %s -nthreads 24 -force' %(join(resultdir,patient,'T1.mif'),
    #                                                        join(resultdir,patient,'5tt_nocoreg.mif'))))
    # if os.path.exists(join(resultdir,patient,'coreg')) == 0:
    #    os.system(str('mkdir %s' %(join(resultdir,patient,'coreg'))))
    # mean_b0_path = join(resultdir,patient,'mean_b0.mif')
    # tt_nocoreg_path = join(resultdir,patient,'5tt_nocoreg.mif')
    # t12dwi_coreg(mean_b0_path, tt_nocoreg_path)
    # os.system(str('5tt2gmwmi %s %s -nthreads 24 -force' %(join(resultdir,patient,'5tt_coreg.mif'),
    #                                                       join(resultdir,patient,'gmwmSeed_coreg.mif'))))
    # os.system(str('tckgen -act %s -backtrack -seed_gmwmi %s -nthreads 24 -maxlength 250 -cutoff 0.06 -select 20000000 %s %s -downsample 3 -force' 
    #               %(join(resultdir,patient,'5tt_coreg.mif'),
    #                 join(resultdir,patient,'gmwmSeed_coreg.mif'),
    #                 join(resultdir,patient,'wmfod_norm.mif'),
    #                 join(resultdir,patient,'tracks_20M.tck'))))
    # os.system(str('tcksift2 -act %s -nthreads 24 %s %s %s -force' %(join(resultdir,patient,'5tt_coreg.mif'),
    #                                                                 join(resultdir,patient,'tracks_20M.tck'),
    #                                                                 join(resultdir,patient,'wmfod_norm.mif'),
    #                                                                 join(resultdir,patient,'sift_20M.txt'))))
    # os.system(str('cp %s %s' %(join(datadir, patient, ses,'anat', str(anat_path)[2:-2]),
    #                            join(resultdir,patient,'coreg','T1.nii.gz'))))
    # track_path = join(resultdir,patient,'tracks_20M.tck')
    # tck2mni_coreg(track_path, mean_b0_path)
    # os.system('rm -r %s' %(track_path))
    # if os.path.exists(join(resultdir,patient,'conn')) == 0:
    #    os.system(str('mkdir %s' %(join(resultdir,patient,'conn'))))
    cmd = [
          'tck2connectome',
          '-symmetric',
          '-zero_diagonal',
          '-scale_invnodevol',
          '-scale_length',
          '-tck_weights_in', join(resultdir, patient,'sift_20M.txt'),
                             join(resultdir, patient,'tracks_20M_MNI.tck'),
                             join(maindir,'Data','ImageDatabase_BIDS','masks','HCP_MMP1_DK','HCP_MMP1_DK_networks.mif'),
                             join(resultdir, patient,'conn','HCP_MMP1_DK_SC_networks_matrix.csv'),
         '-nthreads','24',
         '-out_assignment',join(resultdir,patient,'conn','HCP_MMP1_DK_assignments_networks.csv'),
         '-force'
         ]
    result = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)
    output = result.stdout + result.stderr
    
    # os.system(str('tckedit %s -number 200k %s -force' %(track_path[0:-4] + '_MNI.tck',
    #                                                     track_path[0:-7] + patient + '_200k_MNI.tck')))
    # os.system(str('fsleyes /media/rgong/MRI_BIDS/Data/ImageDatabase_BIDS/masks/MNI152_T1_2mm_neuro.nii.gz --alpha 40 %s'
    #               %(track_path[0:-7] + patient + '_200k_MNI.tck')))

    matches = re.findall(r'\[WARNING\]\s+((?:\d+,\s*)*\d+)', output)
    if matches:
       missing_nodes = []
       for match in matches:
          numbers = [int(n) for n in re.findall(r'\d+', match)]
          missing_nodes.extend(numbers)
          print(f"{patient} — Missing nodes: {missing_nodes}")
          return (patient, missing_nodes)
    else:
      print(f"{patient} — No missing nodes found")
      return (patient,'')
     
def t12dwi_coreg(mean_b0_path, tt_nocoreg_path):

    os.system(str('mrconvert %s %s -nthreads 24 -force' %(mean_b0_path, str(mean_b0_path)[0:-12] + '/coreg/' +'mean_b0.nii.gz')))
    os.system(str('mrconvert %s %s -nthreads 24 -force' %(tt_nocoreg_path, str(tt_nocoreg_path)[0:-15] + '/coreg/' + '5tt_nocoreg.nii.gz')))
    os.system(str('fslroi %s %s 0 1' %(str(tt_nocoreg_path)[0:-15] + '/coreg/' + '5tt_nocoreg.nii.gz',
                                       str(tt_nocoreg_path)[0:-15] + '/coreg/' + '5tt_vol0.nii.gz')))
    os.system(str('flirt -in %s -ref %s -interp nearestneighbour -dof 6 -omat %s' %(str(mean_b0_path)[0:-12] + '/coreg/' +'mean_b0.nii.gz',
                                                                                    str(tt_nocoreg_path)[0:-15] + '/coreg/' + '5tt_vol0.nii.gz',
                                                                                    str(tt_nocoreg_path)[0:-15] + '/coreg/' + 'diff2struct_fsl.mat')))
    os.system(str('transformconvert %s %s %s flirt_import %s -nthreads 24 -force' %(str(tt_nocoreg_path)[0:-15] + '/coreg/' + 'diff2struct_fsl.mat',
                                                                                    str(mean_b0_path)[0:-12] + '/coreg/' +'mean_b0.nii.gz',
                                                                                    str(tt_nocoreg_path)[0:-15] + '/coreg/' + '5tt_nocoreg.nii.gz',
                                                                                    str(tt_nocoreg_path)[0:-15] + '/coreg/' + 'diff2struct_mrtrix.txt')))
    os.system(str('mrtransform %s -linear %s -inverse %s -nthreads 24 -force' %(tt_nocoreg_path, str(tt_nocoreg_path)[0:-15] + '/coreg/' + 'diff2struct_mrtrix.txt', 
                                                                                str(tt_nocoreg_path)[0:-15] + '/coreg/' + '5tt_coreg.mif')))
    os.system(str('cp %s %s' %(str(tt_nocoreg_path)[0:-15] + '/coreg/' + '5tt_coreg.mif', str(tt_nocoreg_path)[0:-15] + '5tt_coreg.mif')))

def tck2mni_coreg(track_path, mean_b0_path):
        
    os.system(str('python /home/rgong/synthstrip-docker -i %s -o %s' %(str(mean_b0_path)[0:-12] + '/coreg/' + 'T1.nii.gz',
                                                                       str(mean_b0_path)[0:-12] + '/coreg/' + 'T1_brain.nii.gz')))
    os.system(str('epi_reg --pedir=-y --echospacing=2.1e-06 --epi=%s --t1=%s --t1brain=%s --out=%s' 
              %(str(mean_b0_path)[0:-12] + '/coreg/' + 'mean_b0.nii.gz',
                str(mean_b0_path)[0:-12] + '/coreg/' + 'T1.nii',
                str(mean_b0_path)[0:-12] + '/coreg/' + 'T1_brain.nii.gz',
                str(mean_b0_path)[0:-12] + '/coreg/' + 'dwi2T1')))
    os.system(str('convert_xfm -omat %s -inverse %s' %(str(mean_b0_path)[0:-12] + '/coreg/' + 'T12dwi.mat',
                                                       str(mean_b0_path)[0:-12] + '/coreg/' + 'dwi2T1.mat')))
    os.system(str('flirt -in %s -ref %s -out %s -init %s -applyxfm' %(str(mean_b0_path)[0:-12] + '/coreg/' + 'T1_brain.nii.gz',
                                                                      str(mean_b0_path)[0:-12] + '/coreg/' +'mean_b0.nii.gz',
                                                                      str(mean_b0_path)[0:-12] + '/coreg/' + 'T12dwi.nii.gz',
                                                                      str(mean_b0_path)[0:-12] + '/coreg/' + 'T12dwi.mat')))
    os.system(str('flirt -ref /media/rgong/MRI_BIDS/Data/ImageDatabase_BIDS/masks/MNI152_T1_2mm_neuro_brain.nii.gz  -in %s -omat %s' 
                                                                             %(str(mean_b0_path)[0:-12] + '/coreg/' + 'T12dwi.nii.gz',
                                                                               str(mean_b0_path)[0:-12] + '/coreg/' + 'T12MNI_affine.mat')))
    os.system(str('fnirt --ref=/media/rgong/MRI_BIDS/Data/ImageDatabase_BIDS/masks/MNI152_T1_2mm_neuro_brain.nii.gz --in=%s --aff=%s --cout=%s'
                                                                              %(str(mean_b0_path)[0:-12] + '/coreg/' + 'T12dwi.nii.gz',
                                                                                str(mean_b0_path)[0:-12] + '/coreg/' + 'T12MNI_affine.mat',
                                                                                str(mean_b0_path)[0:-12] + '/coreg/' + 'warps_T12MNI')))
    os.system(str('invwarp --ref=%s --warp=%s --out=%s' %(str(mean_b0_path)[0:-12] + '/coreg/' + 'T12dwi.nii.gz',
                                                          str(mean_b0_path)[0:-12] + '/coreg/' + 'warps_T12MNI',
                                                          str(mean_b0_path)[0:-12] + '/coreg/' + 'warps_MNI2T1')))
    os.system(str('warpinit -force /media/rgong/MRI_BIDS/Data/ImageDatabase_BIDS/masks/MNI152_T1_2mm_neuro_brain.nii.gz  %s'
                                                                           % str(mean_b0_path)[0:-12] + '/coreg/' + 'inv_identity_warp_no.nii'))
    os.system(str('python /home/rgong/synthstrip-docker -i %s -o %s' %(str(mean_b0_path)[0:-12] + '/coreg/' + 'mean_b0.nii.gz',
                                                                       str(mean_b0_path)[0:-12] + '/coreg/' + 'mean_b0_brain.nii.gz')))
    os.system(str('applywarp --ref=%s --in=%s --warp=%s --out=%s' %(mean_b0_path[0:-12] + '/coreg/' + 'mean_b0_brain.nii.gz',
                                                                    str(mean_b0_path)[0:-12] + '/coreg/' + 'inv_identity_warp_no.nii',
                                                                    str(mean_b0_path)[0:-12] + '/coreg/' + 'warps_MNI2T1.nii.gz',
                                                                    str(mean_b0_path)[0:-12] + '/coreg/' + 'mrtrix_warp_MNI2dwi.nii.gz')))
    os.system(str('mrconvert %s -coord 3 0 - | mrcalc - 0 -mult 1 -force -add %s' % (str(mean_b0_path)[0:-12] + '/coreg/' + 'inv_identity_warp_no.nii',
                                                                                     str(mean_b0_path)[0:-12] + '/coreg/' + 'inv_identity_warp_mask.nii')))
    os.system(str('applywarp --ref=%s --in=%s --warp=%s --out=%s' %(mean_b0_path[0:-12] + '/coreg/' + 'mean_b0_brain.nii.gz', 
                                                                    str(mean_b0_path)[0:-12] + '/coreg/' + 'inv_identity_warp_mask.nii',
                                                                    str(mean_b0_path)[0:-12] + '/coreg/' + 'warps_MNI2T1.nii.gz',
                                                                    str(mean_b0_path)[0:-12] + '/coreg/' + 'mrtrix_warp_MNI2dwi_mask.nii.gz')))
    os.system(str('mrcalc %s 1 nan -force -if %s -mult %s' %(str(mean_b0_path)[0:-12] + '/coreg/' + 'mrtrix_warp_MNI2dwi_mask.nii.gz',
                                                             str(mean_b0_path)[0:-12] + '/coreg/' + 'mrtrix_warp_MNI2dwi.nii.gz',
                                                             str(mean_b0_path)[0:-12] + '/coreg/' + 'mrtrix_warp_MNI2dwi_valid.nii.gz')))
    os.system(str('mrtransform /media/rgong/MRI_BIDS/Data/ImageDatabase_BIDS/masks/MNI152_T1_2mm_neuro_brain.nii.gz -force -warp %s %s' 
                                                          %(str(mean_b0_path)[0:-12] + '/coreg/' + 'mrtrix_warp_MNI2dwi_valid.nii.gz',
                                                            str(mean_b0_path)[0:-12] + '/coreg/' + 'MNI2dwi.nii.gz')))
  #  os.system(str('fsleyes %s' %(str(mean_b0_path)[0:-12] + '/coreg/' + 'MNI2dwi.nii.gz')))
  
    os.system('tcktransform %s %s %s -force -nthreads 24' %(track_path, str(mean_b0_path)[0:-12] + '/coreg/' + 'mrtrix_warp_MNI2dwi_valid.nii.gz',
                                                            track_path[0:-4] + '_MNI.tck'))