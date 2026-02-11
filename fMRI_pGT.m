%% ========================================================================
% Graph-theory analysis 
%% ========================================================================
% set folders and group to analyse
clear
atlas = 'HCP_MMP1_DK';
maindir = pwd;
id = strfind(maindir,'/');
datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Patients','/derivatives','/TS','/',atlas];
maskdir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/masks','/',atlas];
addpath(genpath([maindir(1:id(end) - 1),'/libraries/']));

% obtaining a set of regions used for FC
labels = readtable([maskdir,'/',atlas '_regions.txt']);
labels = table2cell(labels(:,2));
masks = labels;
regions_id = 1:length(masks);

% clinical / demographic metrics
GT_m = readtable([maskdir(1:end - 12),'/','MRI_multicenter_MI_H.xlsx']);
patient_list = table2cell(readtable('fPL_prediction.txt'));
GT_m = table2cell(GT_m);
GT_m = GT_m(ismember(GT_m(:,1),patient_list),:);
GT_m = repelem(GT_m,8,1);

% missing nodes
load('fMN_prediction.mat');
MN = all_missing_nodes;

% community
Ci = table2cell(readtable([maskdir,'/',atlas '_Yeo7.txt']));
Ci = cell2mat(Ci);

%% ========================================================================
% load a FC structure for all participants
load([datadir,'/', atlas '_FC_matrix_prediction.mat']);
for thr = 5:5:25
    for pat = 1:length(group_mat)
        % matrix preparing
        M = group_mat{pat,1};
        M(M < 0) = 0;
        M_thr = M;
       % values = reshape(M,size(M,1)^2,1);
       % values(abs(values) < prctile(abs(values),(100 - thr))) = 0;
       % M_thr = reshape(values,size(M,1),size(M,2));
       % M_thr = abs(M_thr);

        % GT_metrics
        D = degrees_und(M_thr);%/(length(regions_id) - 1); 
        S = strengths_und(M_thr);%/sum(strengths_und(M_thr));
        CCoef = clustering_coef_wu(M_thr);
        CPL = charpath(M_thr);
        BC = betweenness_wei(weight_conversion(M_thr,'lengths')); %/((length(regions_id) - 1)*(length(regions_id) - 2));
        EC = eigenvector_centrality_und(M_thr);
        PC = participation_coef(M_thr, Ci(:,2));
        WM = module_degree_zscore(M_thr, Ci(:,2));
        if find(thr == 5:5:25) == 1
            GT_table(8*(pat-1) + 1, 1) = {'D'};
            GT_table(8*(pat-1) + 2, 1) = {'S'};
            GT_table(8*(pat-1) + 3, 1) = {'CCoef'};
            GT_table(8*(pat-1) + 4, 1) = {'CPL'};
            GT_table(8*(pat-1) + 5, 1) = {'BC'};
            GT_table(8*(pat-1) + 6, 1) = {'EC'};
            GT_table(8*(pat-1) + 7, 1) = {'PC'};
            GT_table(8*(pat-1) + 8, 1) = {'WM'};
        end
        for r = 1:length(masks)
           GT(8*(pat-1) + 1, r) = D(r);  GT(8*(pat-1) + 2, r) = S(r);  GT(8*(pat-1) + 3, r) = CCoef(r); GT(8*(pat-1) + 4, r) = CPL;
           GT(8*(pat-1) + 5, r) = BC(r); GT(8*(pat-1) + 6, r) = EC(r); GT(8*(pat-1) + 7, r) = PC(r);    GT(8*(pat-1) + 8, r) = WM(r);
           if ismember(patient_list(pat),MN(:,1)) == 1
              for node = cell2mat(MN(find(strcmp(patient_list{pat}, MN(:,1))),2))
                  GT(8*(pat-1) + 1:8*(pat-1) + 8, node) = NaN; 
              end
           end
        end
    end
    if find(thr == 5:5:25) == 1
       idx = ~isnan(GT);
       GT_mean = GT;
    else
       idx = ~isnan(GT);
       GT_mean(idx) = GT_mean(idx) + GT(idx);
    end
end
restoredefaultpath; rehash toolboxcache
GT_mean(idx) = GT_mean(idx)/length(5:5:25);
GT_all_table = [GT_m GT_table num2cell(GT_mean)];
GT_all_table = cell2table(GT_all_table,'VariableNames',['ID','Status','Site','Epilepsy Side','Type of surgery','Surgery Side','Gender','Age','AgeOnset',...
                                                        'Duration','Post-surgical duration','Lesion','MTS',...
                                                        'Scalp-EEG bilateral (temporal) spikes','iEEG bilateral (temporal) spikes','Scalp-EEG extratemporal spikes','iEEG extratemporal spikes','EEG spikes', ...
                                                        'iEEG hemisphere','iEEG electrode type','iEEG electrode number','iEEG days of monitoring','iEEG percentage of spikes',...
                                                        'GM volume','WM volume','GT metrics',masks']);
writetable(GT_all_table,[datadir,'/','GT_all_',atlas,'_prediction_mean_wopt.csv']);

%% ========================================================================
% multiple imputation and harmonization
pyenv('Version','/home/rgong/miniconda3/envs/fmri_analysis/bin/python')
py.importlib.import_module('sklearn');
py.importlib.import_module('neuroCombat');
imputer = py.sklearn.impute.KNNImputer(pyargs('n_neighbors',int32(10)));

data_arranged = table2cell(GT_all_table);
data_MI_H = data_arranged;
GT_metrics = {'D','S','CCoef','CPL','BC','EC','PC','WM'};
for GT_metric = GT_metrics
    GT_data = data_arranged(strcmp(data_arranged(:,26),GT_metric),[24,25,27:end])';
    GT_data = cell2mat(GT_data);
    GT_data_py = py.numpy.array(GT_data);
    GT_data_filled_py = imputer.fit_transform(GT_data_py);
    GT_data_filled = double(GT_data_filled_py)';
    
    covars = data_arranged(strcmp(data_arranged(:,26),GT_metric),[3,6])';
    covars_list = py.dict(pyargs('Site',covars(1,:),'Side',covars(2,:)));

    py.importlib.import_module('pandas');
    covars_py = py.pandas.DataFrame(covars_list);
    batch_col = py.list({'Site'});

    combat_result = py.neuroCombat.neuroCombat(pyargs('dat',GT_data_filled_py,'covars',covars_py,'batch_col',batch_col));
    GT_data_combat_py = combat_result{'data'};
    GT_data_combat_np = py.numpy.round(GT_data_combat_py,int32(5));
    GT_data_combat = double(GT_data_combat_np)';
    data_MI_H(strcmp(data_MI_H(:,26),GT_metric),[24,25,27:end]) = num2cell(GT_data_combat);
end
GT_all_table_MI_H = cell2table(data_MI_H,'VariableNames',['ID','Status','Site','Epilepsy Side','Type of surgery','Surgery Side','Gender','Age','AgeOnset',...
                                                          'Duration','Post-surgical duration','Lesion','MTS',...
                                                          'Scalp-EEG bilateral (temporal) spikes','iEEG bilateral (temporal) spikes','Scalp-EEG extratemporal spikes','iEEG extratemporal spikes','EEG spikes',...
                                                          'iEEG hemisphere','iEEG electrode type','iEEG electrode number','iEEG days of monitoring','iEEG percentage of spikes',...
                                                           'GM volume','WM volume','GT metrics',masks']);
writetable(GT_all_table_MI_H,[datadir,'/','GT_all_',atlas,'_prediction_mean_wopt_MI_H.csv']);

%% ========================================================================
% atypical hubness
data_MI_H_normative = table2cell(readtable([maindir(1:id(end) - 15),'/cMRI_BIDS','/Data','/ImageDatabase_BIDS','/Controls',...
                                                                    '/derivatives','/TS','/HCP_MMP1_DK','/GT_all_HCP_MMP1_DK_ahubness_mean_wnc_MI_H.csv']));
data_MI_H_atypical = table2cell(GT_all_table_MI_H);
GT_metrics = {'D','S','CCoef','CPL','BC','EC','PC','WM'};
for GT_metric = GT_metrics
    GT_data_normative = data_MI_H_normative(strcmp(data_MI_H_normative(:,7),GT_metric),[5,6,8:end]);
    GT_data_normative = cell2mat(GT_data_normative);
    GT_data_atypical = data_MI_H_atypical(strcmp(data_MI_H_atypical(:,26),GT_metric),[24,25,27:end]);
    GT_data_atypical = cell2mat(GT_data_atypical);
    GT_data_atypical_z_score = [];
    for i = 1:length(GT_data_atypical)
        GT_data_atypical_z_score(:,i) = (GT_data_atypical(:,i) - mean(GT_data_normative(:,i)))/std(GT_data_normative(:,i));
    end
    data_MI_H_atypical(strcmp(data_MI_H_atypical(:,26),GT_metric),[24,25,27:end]) = num2cell(GT_data_atypical_z_score);
    clear GT_data_atypical_z_score
end
GT_all_table_MI_H_atypical = cell2table(data_MI_H_atypical,'VariableNames',['ID','Status','Site','Epilepsy Side','Type of surgery','Surgery Side','Gender','Age','AgeOnset',...
                                                                            'Duration','Post-surgical duration','Lesion','MTS',...
                                                                            'Scalp-EEG bilateral (temporal) spikes','iEEG bilateral (temporal) spikes','Scalp-EEG extratemporal spikes','iEEG extratemporal spikes','EEG spikes',...
                                                                            'iEEG hemisphere','iEEG electrode type','iEEG electrode number','iEEG days of monitoring','iEEG percentage of spikes',...
                                                                            'GM volume','WM volume','GT metrics',masks']);
writetable(GT_all_table_MI_H_atypical,[datadir,'/','GT_all_',atlas,'_prediction_mean_wnc_MI_H_atypical.csv']);

%% ========================================================================
% ispi /contralateral assignment
if strcmp(atlas,'Schaefer400_7N') == 1
   cols = masks(1:length(masks)/2);
   cols = regexprep(cols,'LH_','');
elseif strcmp(atlas,'AICHA') == 1
    cols = masks(1:2:end);
    cols = regexprep(cols,'-L','')';
elseif strcmp(atlas,'AAL') == 1
    cols = masks(endsWith(masks, '_L'));
    cols = cellfun(@(x) extractBefore(x, strlength(x) - 1),cols,'UniformOutput',false);
elseif strcmp(atlas,'HCP_MMP1_DK') == 1
    cols = [masks(1:180);masks(361:367)];
    cols = cellfun(@(x) extractAfter(x, 2),cols,'UniformOutput',false);
end
region_names = cols;
data = GT_all_table_MI_H_atypical;
new_data = data(:,1:26);
for i = 1:length(region_names)
    region = region_names{i};
    ipsi_values = zeros(height(data), 1);
    contra_values = zeros(height(data), 1);
    for j = 1:height(data)
        surgery_side = data.('Surgery Side'){j};
        if strcmp(atlas,'Schaefer400_7N') == 1
            L_col = ['LH_' region];
            R_col = ['RH_' region];
        elseif strcmp(atlas,'AICHA') == 1
            L_col = [region '-L'];
            R_col = [region '-R'];
        elseif strcmp(atlas,'AAL') == 1
            L_col = [region '_L'];
            R_col = [region '_R'];
        elseif strcmp(atlas,'HCP_MMP1_DK') == 1
            L_col = ['L_' region];
            R_col = ['R_' region];
        end
        L_value = data.(L_col)(j);
        R_value = data.(R_col)(j);
        % Assign values
        if strcmp(surgery_side, 'L')
            ipsi_values(j) = L_value;
            contra_values(j) = R_value;
        elseif strcmp(surgery_side, 'R')
            ipsi_values(j) = R_value;
            contra_values(j) = L_value;
        end
    end
    ipsi_colname = ['ipsi_' region];
    contra_colname = ['contra_' region];
    new_data.(ipsi_colname) = ipsi_values;
    new_data.(contra_colname) = contra_values;
end
cols = new_data.Properties.VariableNames;
nonregion_cols = cols(1:26);
I_cols = cols(contains(cols,'ipsi_'));
C_cols = cols(contains(cols,'contra_'));
new_order = [nonregion_cols, I_cols, C_cols];
data_arranged = new_data(:,new_order);
writetable(data_arranged,[datadir,'/GT_all_',atlas,'_prediction_mean_wnc_MI_H_atypical_ipsicontra.csv']);