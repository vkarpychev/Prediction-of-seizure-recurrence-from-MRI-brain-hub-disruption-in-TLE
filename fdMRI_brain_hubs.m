%% ========================================================================
% folders
clear
atlas = 'HCP_MMP1_DK';
maindir = pwd;
id = strfind(maindir,'/');
maskdir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/masks','/',atlas];
datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Controls'];
addpath(genpath([maindir(1:id(end) - 6),'/code','/libraries/']));

% obtaining a set of regions
labels = readtable([maskdir,'/',atlas '_regions.txt']);
labels = table2cell(labels(:,2));
masks = labels;
regions_id = 1:length(masks);

% temporal lobe ROIs
TL = {'TG','TE','TF','STS','STG','MTG','ITG','TPOJ', ...                        
      'A1','LBelt','MBelt','PBelt','RI','EC','PeEc','PHA','PH', ...        
      'amygdala','hippocampus','thalamus'};
TL_masks = false(size(masks));
for r = 1:numel(TL)
    TL_masks = TL_masks | contains(masks, TL(r));
end
TL_idx = find(TL_masks);

%% ========================================================================
for modality = {'SC','FC'}
    if strcmp(modality,'SC') == 1
        folder = 'MRtrix';
    elseif strcmp(modality,'FC') == 1  
        folder = ['TS/',atlas];
    end
    GT_datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Controls','/derivatives','/',folder];
    GT_T = readtable([GT_datadir,'/','GT_all_' atlas '_ahubness_mean_MI_H.csv']);
    control_list = unique(GT_T.ID);
    % TL_S_GT = GT_T(strcmp(GT_T{:,7},'TL_S'),8:end);
    % TL_S_hubs = zeros(1,length(table2array(TL_S_GT)));
    % TL_S_hubs(mean(table2array(TL_S_GT)) >= prctile(mean(table2array(TL_S_GT)),20)) = 1;
    % for roi = 1:length(masks)
    %    if TL_S_hubs(roi) == 0 && ismember(roi,TL_idx) == 1
    %      TL_S_hubs(roi) =  TL_S_hubs(roi) + 1;
    %    end
    % end
    
    BC_GT = GT_T(strcmp(GT_T{:,7},'BC'),8:end);
    BC_hubs = zeros(1,length(table2array(BC_GT)));
    BC_hubs_mean = mean(table2array(BC_GT));
    BC_hubs(BC_hubs_mean >= prctile(BC_hubs_mean,80)) = 1;

    EC_GT = GT_T(strcmp(GT_T{:,7},'EC'),8:end);
    EC_hubs = zeros(1,length(table2array(EC_GT)));
    EC_hubs_mean = mean(table2array(EC_GT));
    EC_hubs(EC_hubs_mean >= prctile(EC_hubs_mean,80)) = 1;

    S_GT = GT_T(strcmp(GT_T{:,7},'S'),8:end);
    S_hubs = zeros(1,length(table2array(S_GT)));
    S_hubs_mean = mean(table2array(S_GT));
    S_hubs(S_hubs_mean >= prctile(S_hubs_mean,80)) = 1;

    CPL_GT = GT_T(strcmp(GT_T{:,7},'CPL'),8:end);
    CPL_hubs = zeros(1,length(table2array(CPL_GT)));
    CPL_hubs_mean = mean(table2array(CPL_GT));
    CPL_hubs(CPL_hubs_mean <= prctile(CPL_hubs_mean,20)) = 1;

    CCoef_GT = GT_T(strcmp(GT_T{:,7},'CCoef'),8:end);
    CCoef_hubs = zeros(1,length(table2array(CCoef_GT)));
    CCoef_hubs_mean = mean(table2array(CCoef_GT));
    CCoef_hubs(CCoef_hubs_mean <= prctile(CCoef_hubs_mean,20)) = 1;

    PC_GT = GT_T(strcmp(GT_T{:,7},'PC'),8:end);
    PC_hubs = zeros(1,length(table2array(PC_GT)));
    PC_hubs_mean = mean(table2array(PC_GT));
    PC_hubs(PC_hubs_mean >= prctile(PC_hubs_mean,80)) = 1;

    D_GT = GT_T(strcmp(GT_T{:,7},'D'),8:end);
    D_hubs = zeros(1,length(table2array(D_GT)));
    D_hubs_mean = mean(table2array(D_GT));
    D_hubs(D_hubs_mean >= prctile(D_hubs_mean,20)) = 1;
  
    BC_final_hubs = BC_hubs + S_hubs + CPL_hubs; 
    EC_final_hubs = EC_hubs + S_hubs + CPL_hubs;
    PC_final_hubs = PC_hubs + D_hubs;
   
    %% ====================================================================
    % all hubs across the metrics
    display(sum(BC_final_hubs >= 2));
    save(sprintf('%s_BC_brain_hubs.mat',char(modality)),'BC_final_hubs');
    display(sum(EC_final_hubs >= 2));
    save(sprintf('%s_EC_brain_hubs.mat',char(modality)),'EC_final_hubs');
    display(sum(PC_final_hubs >= 2));
    save(sprintf('%s_PC_brain_hubs.mat',char(modality)),'PC_final_hubs');
end