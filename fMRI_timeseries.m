%% ========================================================================
% exctracting timeseries(ts) based on 3dmaskdump AFNI command from fMRI
% initiation of directories

clear 
maindir = pwd;
id = strfind(maindir,'/');
datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Patients','/derivatives','/fmriprep'];
maskdir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/masks','/HCP_MMP1_DK','/HCP_MMP1_DK_regions'];
afni_command = '/home/lambda/abin/3dmaskdump';

folder = dir([datadir,'/','sub-*']);
folder = folder(~endsWith(string({folder.name}),'.html'));
fpatient_list = table2cell(readtable('fpatient_list.txt'));
folder = folder(ismember({folder.name},fpatient_list));
folder = folder([folder(:).isdir]);
session = 'ses-pre';
masks = readtable([maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/masks','/HCP_MMP1_DK','/HCP_MMP1_DK_regions.txt']);
masks = table2cell(masks(:,2));
all_missing_nodes = {};
% running the process
for crun = 1:length(folder)
   try
      fmri_image = dir([datadir,'/',folder(crun).name,'/',session,'/func/','*smooth_bold.nii.gz']);
      output_folder = [maindir(1:id(end) - 6) '/Data','/ImageDatabase_BIDS','/Patients','/derivatives',...
                                '/TS','/HCP_MMP1_DK','/',folder(crun).name]; 
       if isfolder(output_folder) == 0    
          mkdir(output_folder);
       end
       % for each participant, saving ts across all rois
       if isfile([output_folder,'/','roi_ts.txt']) == 0
          for roi = 1:length(masks)
              cmd = sprintf('%s -noijk -mask %s %s > %s',afni_command,[maskdir filesep [char(masks(roi)) '.nii']],...
                                         [datadir filesep folder(crun).name filesep session filesep 'func' filesep fmri_image.name],...
                                         [output_folder filesep 'ts.txt']);
              system(cmd);
              ts = importdata([output_folder filesep 'ts.txt']);
              if roi == 1
                 roi_ts = mean(ts(any(ts,2),:),1)';
              else
                 roi_ts = [roi_ts mean(ts(any(ts,2),:),1)'];
              end
              if all(isnan(roi_ts(:,roi))) == 1
                 roi_ts(isnan(roi_ts(:,roi)),roi) = 0;
                 if exist('missing_nodes', 'var') == 0
                     missing_nodes = roi;
                 else
                     missing_nodes = [missing_nodes roi];
                 end
              end
          end
          restoredefaultpath; rehash toolboxcache
          writetable(array2table(roi_ts,'VariableNames',masks),[output_folder,'/','roi_ts.txt']);
          if isfile([output_folder,'/','ts.txt']) == 1
             delete([output_folder,'/','ts.txt']);
          end
       else
          roi_ts = importdata([output_folder,'/','roi_ts.txt']);
          roi_ts = roi_ts.data;
          if isfile([output_folder,'/','ts.txt']) == 1
             delete([output_folder,'/','ts.txt']);
          end
       end
       % writing a cell structure for cPPI
       if exist('missing_nodes', 'var') == 1
          all_missing_nodes{end + 1,1} = folder(crun).name;
          all_missing_nodes{end,2} = unique(missing_nodes);
       end
       timecourse_cell{crun,1} = roi_ts;
       cd(maindir);
   catch 
         disp(['Error with ',folder(crun).name]);
   end
   clear missing_nodes roi_ts ts roi output_folder cmd fmri_image
end
timecourse_cell = timecourse_cell(~ cellfun('isempty',timecourse_cell));
save('fMRI_missing_nodes.mat','all_missing_nodes');
save([maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Patients','/derivatives',...
                             '/TS','/HCP_MMP1_DK','/timecourse_cell.mat'],'timecourse_cell');
