%% ========================================================================
% Graph-theory analysis 
%% ========================================================================
% set folders and group to analyse
clear
atlas = 'HCP_MMP1_DK';
maindir = pwd;
id = strfind(maindir,'/');
datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Controls','/derivatives','/TS','/',atlas];
maskdir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/masks','/',atlas];
addpath(genpath([maindir(1:id(end) - 1),'/libraries/']));

% obtaining a set of regions used for FC
labels = readtable([maskdir,'/',atlas '_regions.txt']);
labels = table2cell(labels(:,2));
masks = labels;
regions_id = 1:length(masks);

% clinical / demographic metrics
GT_m = readtable([maskdir(1:end - 12),'/','fsCL_multicenter_MI_H.xlsx']);
patient_list = table2cell(readtable('fCL_ahubness.txt'));
GT_m = table2cell(GT_m);
GT_m = GT_m(ismember(GT_m(:,1),patient_list),:);
GT_m = repelem(GT_m,10,1);

% missing nodes
load('fMN_ahubness.mat');
MN = all_missing_nodes;

% community
Ci = table2cell(readtable([maskdir,'/',atlas '_Yeo7.txt']));
Ci = cell2mat(Ci);
TL = {'TG','TE','TF','STS','STG','MTG','ITG','TPOJ', ...                        
      'A1','LBelt','MBelt','PBelt','RI','EC','PeEc','PHA','PH', ...        
      'amygdala','hippocampus','thalamus'};
TL_masks = false(size(masks));
for r = 1:numel(TL)
    TL_masks = TL_masks | contains(masks, TL(r));
end
TL_idx = find(TL_masks);

%% ========================================================================
% load a FC structure for all participants
load([datadir,'/', atlas '_FC_matrix_ahubness.mat']);
for thr = 5:5:25
    for pat = 1:length(group_mat)
        % matrix preparing
        M = group_mat{pat,1};
        M(M < 0) = 0;
       % values = reshape(M,size(M,1)^2,1);
       % values(abs(values) < prctile(abs(values),(100 - thr))) = 0;
       % M_thr = reshape(values,size(M,1),size(M,2));
       % M_thr = abs(M_thr);
        M_thr = M;
        
        % GT_metrics
        D = degrees_und(M_thr); %/(length(regions_id) - 1); 
        S = strengths_und(M_thr);
        CCoef = clustering_coef_wu(M_thr);
        BC = betweenness_wei(weight_conversion(M_thr,'lengths')); % /((length(regions_id) - 1)*(length(regions_id) - 2));
        EC = eigenvector_centrality_und(M_thr);
        PC = participation_coef(M_thr, Ci(:,2));
        WM = module_degree_zscore(M_thr, Ci(:,2));
        GE_wb = efficiency_wei(M_thr);
        Distance = distance_wei(weight_conversion(M_thr,'lengths'));
        CPL = nan(size(M_thr,1),1);
        for i = 1:size(M_thr,1)
            di = Distance(i,:);
            di(i) = NaN;                    
            di(isinf(di)) = NaN;             
            CPL(i) = mean(di,'omitnan'); % / (length(regions_id) - 1);    
        end
        TL_S = sum(M_thr(:,TL_idx),2)';
        if find(thr == 5:5:25) == 1
            GT_table(10*(pat-1) + 1, 1) = {'D'};
            GT_table(10*(pat-1) + 2, 1) = {'S'};
            GT_table(10*(pat-1) + 3, 1) = {'CCoef'};
            GT_table(10*(pat-1) + 4, 1) = {'BC'};
            GT_table(10*(pat-1) + 5, 1) = {'EC'};
            GT_table(10*(pat-1) + 6, 1) = {'PC'};
            GT_table(10*(pat-1) + 7, 1) = {'WM'};
            GT_table(10*(pat-1) + 8, 1) = {'CPL'};
            GT_table(10*(pat-1) + 9, 1) = {'GE_wb'};
            GT_table(10*(pat-1) + 10, 1) = {'TL_S'};
        end
        for r = 1:length(masks)
           GT(10*(pat-1) + 1, r) = D(r);   GT(10*(pat-1) + 2, r) = S(r);  GT(10*(pat-1) + 3, r) = CCoef(r); 
           GT(10*(pat-1) + 4, r) = BC(r);  GT(10*(pat-1) + 5, r) = EC(r); GT(10*(pat-1) + 6, r) = PC(r); GT(10*(pat-1) + 7, r) = WM(r);
           GT(10*(pat-1) + 8, r) = CPL(r); GT(10*(pat-1) + 9, r) = GE_wb; GT(10*(pat-1) + 10, r) = TL_S(r);
           if ismember(patient_list(pat),MN(:,1)) == 1
              for node = cell2mat(MN(find(strcmp(patient_list{pat},MN(:,1))),2))
                  GT(10*(pat-1) + 1:10*(pat-1) + 10, node) = NaN; 
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
GT_all_table = cell2table(GT_all_table,'VariableNames',['ID','Site','Gender','Age','GM volume','WM volume','GT_metrics',masks']);
writetable(GT_all_table,[datadir,'/','GT_all_',atlas,'_ahubness_mean_wpt.csv']);

%% ========================================================================
% multiple imputation and harmonization
pyenv('Version','/home/rgong/miniconda3/envs/fmri_analysis/bin/python3.9')
py.importlib.import_module('sklearn');
py.importlib.import_module('neuroCombat');
imputer = py.sklearn.impute.KNNImputer(pyargs('n_neighbors',int32(10)));

data_arranged = table2cell(GT_all_table);
data_MI_H = data_arranged;
GT_metrics = {'D','S','CCoef','BC','EC','PC','WM','CPL','TL_S'};
for GT_metric = GT_metrics
    GT_data = data_arranged(strcmp(data_arranged(:,7),GT_metric),8:end)';
    GT_data = cell2mat(GT_data);
    GT_data_py = py.numpy.array(GT_data);
    GT_data_filled_py = imputer.fit_transform(GT_data_py);
    GT_data_filled = double(GT_data_filled_py)';
    
    covars = data_arranged(strcmp(data_arranged(:,7),GT_metric),2:4)';
    sites = tabulate(covars(1,:));
    count_sites = sites(cell2mat(sites(:,2)) <= 3, 1);
    for i = 1:length(count_sites)
        site_replaced = count_sites{i};
        covars(1,strcmp(covars(1,:),site_replaced)) = {'Other'};
    end
    covars_list = py.dict(pyargs('Site',covars(1,:),'Gender',covars(2,:),'Age',covars(3,:)));

    py.importlib.import_module('pandas');
    covars_py = py.pandas.DataFrame(covars_list);
    batch_col = py.list({'Site'});

    combat_result = py.neuroCombat.neuroCombat(pyargs('dat',GT_data_filled_py,'covars',covars_py,'batch_col',batch_col));
    GT_data_combat_py = combat_result{'data'};
    GT_data_combat_np = py.numpy.round(GT_data_combat_py,int32(5));
    GT_data_combat = double(GT_data_combat_np)';
    data_MI_H(strcmp(data_MI_H(:,7),GT_metric),8:end) = num2cell(GT_data_combat);
end
GT_all_table_MI_H = cell2table(data_MI_H,'VariableNames',['ID','Site','Gender','Age','GM volume','WM volume','GT_metrics',masks']);
writetable(GT_all_table_MI_H,[datadir,'/','GT_all_',atlas,'_ahubness_mean_wpt_MI_H.csv']);