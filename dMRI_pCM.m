%% ========================================================================
% Postpocessing of structural connectivity matrices

clear
atlas = 'HCP_MMP1_DK';
maindir = pwd;
id = strfind(maindir,'/');
datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Patients','/derivatives','/MRtrix'];
maskdir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/masks','/',atlas];
labels = readtable([maskdir,'/',atlas '_regions.txt']);
labels = table2cell(labels(:,2));

folder = dir([datadir,'/','sub*']);
patient_list = table2cell(readtable('dPL_prediction.txt'));
folder = folder(ismember({folder.name},patient_list));
folder = folder([folder(:).isdir]);

for i = 1:length(folder)
    M = table2cell(readtable([folder(i).folder,'/',folder(i).name,'/conn','/',atlas '_SC_matrix.csv']));
    M = cell2mat(M);
    idx = M > 0 & ~isnan(M);
    M_log = M;
    M_log(idx) = log(M_log(idx));
    M_log(isnan(M_log)) = 0;
    group_mat{i,1} = M_log;
    % system(sprintf('antsRegistrationSyNQuick.sh -d 3 -f %s -m %s -o %s',...
    %                 [datadir(1:end - 28),'/masks','/MNI152_T1_2mm_neuro_brain.nii.gz'], ...
    %                 [datadir,'/',folder(i).name,'/coreg','/mean_b0_brain.nii.gz'], ...
    %                 [datadir,'/',folder(i).name,'/coreg','/mean_b02MNI']));
    % system(sprintf('antsApplyTransforms -d 3 -i %s -r %s -t %s -t %s -o %s', ...
    %                 [datadir,'/',folder(i).name,'/coreg','/mean_b0_brain.nii.gz'], ...
    %                 [datadir(1:end - 28),'/masks','/MNI152_T1_2mm_neuro_brain.nii.gz'], ...
    %                 [datadir,'/',folder(i).name,'/coreg','/mean_b02MNI1Warp.nii.gz'], ...
    %                 [datadir,'/',folder(i).name,'/coreg','/mean_b02MNI0GenericAffine.mat'], ...
    %                 [datadir,'/',folder(i).name,'/coreg','/mean_b02MNI.nii']));
    % system(sprintf('fslmaths %s -thr 0 %s', ...
    %                 [datadir,'/',folder(i).name,'/coreg','/mean_b02MNI.nii'], ...
    %                 [datadir,'/',folder(i).name,'/coreg','/mean_b02MNI_thr.nii']));
    % excluded_regions = {};
    % for r = 1:length(labels)
    %     rpath = [maskdir,'/',atlas '_regions','/',char(labels(r)),'.nii'];
    %     system(sprintf('fslmaths %s -mul %s %s',rpath, ...
    %                 [datadir,'/',folder(i).name,'/coreg','/mean_b02MNI_thr.nii'], ...
    %                 [datadir,'/',folder(i).name,'/coreg','/',char(labels(r)) '_overlap.nii']))
    %     [~,output_1] = system(sprintf('fslstats %s -V', rpath));
    %     rvoxels = str2double(strtok(output_1));
    %     [~,output_2] = system(sprintf('fslstats %s -V',[datadir,'/',folder(i).name,'/coreg','/',char(labels(r)) '_overlap.nii']));
    %     ovoxels = str2double(strtok(output_2));
    %     perc_overlap = ovoxels / rvoxels;
    %     if perc_overlap < 0.3
    %         index = find(strcmp(labels(r),labels));
    %         M(index,:) = NaN;
    %         M(:,index) = NaN;
    %         excluded_regions = [excluded_regions; labels(r)];
    %     end
    %     system(sprintf('rm -r %s',[datadir,'/',folder(i).name,'/coreg','/',char(labels(r)) '_overlap.nii.gz']))
    % end
    % M_log = M;
    % idx = M > 0 & ~isnan(M);
    % M_log(idx) = log(M(idx));
    % M_log = triu(M_log);
    % % mice [python]
    % py.importlib.import_module('fancyimpute');
    % knn = py.fancyimpute.SoftImpute();
    % M_log_py = py.numpy.array(M_log);
    % M_log_filled_py = knn.fit_transform(M_log_py);
    % M_log_filled = double(M_log_filled_py);
    % 
    % group_mat{i,1} = M_log;
    % excluded_regions_all{i,1} = excluded_regions;
end
save([maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Patients','/derivatives',...
                              '/MRtrix','/',atlas '_SC_matrix_prediction.mat'],'group_mat');
