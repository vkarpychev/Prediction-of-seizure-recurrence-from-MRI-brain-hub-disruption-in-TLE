#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:55:18 2025

@author: rgong
"""

import os
from os.path import join
from multiprocessing import Pool


maindir = os.path.dirname(os.path.split(os.getcwd())[0])
datadir = join(maindir,'Data/ImageDatabase_BIDS/Patients/derivatives/fmriprep')
ses = 'ses-pre'
os.system(str('echo "zekelab_ieeg" | sudo -S usermod -aG docker rgong'))
os.system(str('newgrp docker'))

with open('fPL_ahubness.txt', 'r') as file:
    fpatient_list = [line.strip() for line in file.readlines() if line.strip()]
    del fpatient_list[0]
    
with open('dPL_ahubness.txt', 'r') as file:
    dpatient_list = [line.strip() for line in file.readlines() if line.strip()]
    del dpatient_list[0]
    
patient_list = list(set(fpatient_list + dpatient_list))
  
patient_list_upd = []
for participant in patient_list:
    if os.path.isfile(join(datadir,'%s.html' %(participant))) == 0:
        print(f"Still processing: {participant}")
        patient_list_upd.append(participant)
                
chunk_size = 5
n_chunks = (len(patient_list_upd) + chunk_size - 1) // chunk_size

chunks = []
for i in range(n_chunks):
   start = i * chunk_size
   end = min(start + chunk_size, len(patient_list_upd))
   chunks.append(patient_list_upd[start:end])
   
def run_fmriprep(chunk):
   participants = ' '.join(chunk)
   command = (str('echo "zekelab_ieeg" | sudo -S docker run -i --rm '
                  '-v /media/rgong/MRI_BIDS/Data/ImageDatabase_BIDS/Patients:/data:ro '
                  '-v /media/rgong/MRI_BIDS/Data/ImageDatabase_BIDS/Patients/derivatives/fmriprep:/out '
                  '-v /media/rgong/MRI_BIDS/Data/ImageDatabase_BIDS/work:/work '
                  '-v /media/rgong/MRI_BIDS/Data/ImageDatabase_BIDS/fmri_bids_filter.json:/tmp/filter.json '
                  '-v /usr/local/freesurfer/7.3.2/license.txt:/usr/local/freesurfer/7.3.2/license.txt '
                  'nipreps/fmriprep:latest /data /out participant --work-dir /work '
                  '--output-spaces MNI152NLin6Asym:res-2 --ignore slicetiming t2w --fs-no-reconall '
                  '--skip_bids_validation --fs-license-file /usr/local/freesurfer/7.3.2/license.txt '
                  '--bids-filter-file /tmp/filter.json --nthreads 16 --mem_mb 48000 '
                  '--participant-label %s' %(participants)))
   return os.system(command)    

if __name__ == '__main__':
   with Pool(processes = 1) as pool:
       results = pool.map(run_fmriprep, chunks)
       
       
       
       