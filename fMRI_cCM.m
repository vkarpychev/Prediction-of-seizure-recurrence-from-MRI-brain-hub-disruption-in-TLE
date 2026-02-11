%% ========================================================================
% Functional connectivity analysis 
%% ========================================================================
% set folders and group to analyse
clear
maindir = pwd;
id = strfind(maindir,'/');
datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Controls','/derivatives','/TS','/HCP_MMP1_DK'];
% defining time-series data
TS = dir([datadir,'/TC_cell_ahubness.mat']);
% loading time-series data
load([datadir,'/',TS.name]);

% counting connectivity matrices
for i = 1:length(timecourse_cell)
    pat_ts = cell2mat(timecourse_cell(i,1));
    source_region = 1:size(pat_ts,2);
    while isempty(source_region) == 0
        for target_region = source_region(1:end)
            coeff = corrcoef(pat_ts(:,source_region(1)),pat_ts(:,target_region));
            pat_mat(source_region(1),target_region) = coeff(1,2);
        end
        source_region(1) = [];
        clear coeff
    end
    pat_mat = pat_mat - diag(diag(pat_mat));
    pat_mat(isnan(pat_mat)) = 0;
    idx = ~ logical(eye(size(pat_mat)));
    pat_mat(idx) = atanh(pat_mat(idx));
    pat_mat = pat_mat + pat_mat';
    group_mat{i,1} = pat_mat;
    clear pat_ts source_region target_region pat_mat
end
save([maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Controls','/derivatives',...
                             '/TS', '/HCP_MMP1_DK','/HCP_MMP1_DK_FC_matrix_ahubness.mat'],'group_mat');