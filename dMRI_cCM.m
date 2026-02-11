%% ========================================================================
% Postpocessing of structural connectivity matrices
clear
atlas = 'HCP_MMP1_DK';
maindir = pwd;
id = strfind(maindir,'/');
datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Controls','/derivatives','/MRtrix'];
maskdir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/masks','/',atlas];
labels = readtable([maskdir,'/',atlas '_regions.txt']);
labels = table2cell(labels(:,2));
folder = dir([datadir,'/','sub*']);
patient_list = table2cell(readtable('dCL_ahubness.txt'));
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
end
save([maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Controls','/derivatives',...
                              '/MRtrix','/',atlas '_SC_matrix_ahubness.mat'],'group_mat');