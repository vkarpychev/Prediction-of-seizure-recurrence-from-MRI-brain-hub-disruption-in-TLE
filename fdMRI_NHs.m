%% ========================================================================
% folders
clear
atlas = 'HCP_MMP1_DK';
maindir = pwd;
id = strfind(maindir,'/');
maskdir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/masks','/',atlas];
datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Patients'];
addpath(genpath([maindir(1:id(end) - 6),'/code','/libraries/']));
Ci = table2cell(readtable([maskdir,'/',atlas '_Yeo7.txt']));
Ci = cell2mat(Ci(:,2));

%% ========================================================================
% comparisons with healthy controls
for modality = {'SC','FC'}
    if strcmp(modality,'SC') == 1
        folder = 'MRtrix';
    elseif strcmp(modality,'FC') == 1  
        folder = ['TS/',atlas];
    end
    GT_datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Patients','/derivatives','/',folder];
    GT_T = readtable([GT_datadir,'/','GT_all_' atlas '_prediction_mean_MI_H_atypical_ipsicontra.csv']);
    regions = GT_T.Properties.VariableNames(27:end);
    GT_T = GT_T(~ strcmp(GT_T.Site,'Emory'),:);
    patient_list = unique(GT_T.ID);
    for GT_metric = {'PC'}
        GT_data = table2cell(GT_T(strcmp(GT_T{:,26},GT_metric),[2,27:end]));
        GT_SF = cell2mat(GT_data(strcmp(GT_data(:,1),'SF'),2:end));
        GT_NSF = cell2mat(GT_data(strcmp(GT_data(:,1),'NSF'),2:end));
        if strcmp(GT_metric,'BC') == 1
            load([maindir(1:id(end) - 15),'/cMRI_BIDS','/code','/fs_ahubness','/',sprintf('%s_BC_brain_hubs.mat',char(modality))]);
            BC_final_hubs(BC_final_hubs < 2) = 0; BC_final_hubs(BC_final_hubs >= 2) = 1;
            brain_hubs = BC_final_hubs;
        elseif strcmp(GT_metric,'EC') == 1
            load([maindir(1:id(end) - 15),'/cMRI_BIDS','/code','/fs_ahubness','/',sprintf('%s_EC_brain_hubs.mat',char(modality))]);
            EC_final_hubs(EC_final_hubs < 2) = 0; EC_final_hubs(EC_final_hubs >= 2) = 1;
            brain_hubs = EC_final_hubs;
        elseif strcmp(GT_metric,'PC') == 1
            load([maindir(1:id(end) - 15),'/cMRI_BIDS','/code','/fs_ahubness','/',sprintf('%s_PC_brain_hubs.mat',char(modality))]);
            PC_final_hubs(PC_final_hubs < 2) = 0; PC_final_hubs(PC_final_hubs >= 2) = 1;
            brain_hubs = PC_final_hubs;
        end
        symmetrical_brain_hubs = zeros(size(brain_hubs));
        symmetrical_brain_hubs(1:180) = brain_hubs(1:180) & brain_hubs(181:360); 
        symmetrical_brain_hubs(181:360) = brain_hubs(1:180) & brain_hubs(181:360); 
        symmetrical_brain_hubs(361:367) = brain_hubs(361:367) & brain_hubs(368:end); 
        symmetrical_brain_hubs(368:end) = brain_hubs(361:367) & brain_hubs(368:end); 
        display(sum(symmetrical_brain_hubs));
        symmetrical_brain_hubs = [symmetrical_brain_hubs(1:180) symmetrical_brain_hubs(361:367) symmetrical_brain_hubs(181:360) symmetrical_brain_hubs(368:end)];
        GT_SF(:,symmetrical_brain_hubs ~= 1) = 0;
        [~,p_SF,~,stats_SF] = ttest(GT_SF);
        t_SF = stats_SF.tstat;
        t_SF(isnan(t_SF)) = 0;
        p_fdr_SF = zeros(size(p_SF));
        p_fdr_SF(symmetrical_brain_hubs == 1) = mafdr(p_SF(symmetrical_brain_hubs == 1), 'BHFDR', true);
        t_fdr_SF = zeros(1,length(t_SF));
        t_fdr_SF(p_fdr_SF < 0.05) = t_SF(p_fdr_SF < 0.05);
        t_SF([184,371]) = t_SF([120,307]);
        t_SF([120,307]) = 0;
        t_fdr_SF([184,371]) = t_fdr_SF([120,307]);
        t_fdr_SF([120,307]) = 0;
        display(regions(abs(t_fdr_SF) > 0));
        display(t_fdr_SF(abs(t_fdr_SF) > 0));

        GT_NSF(:,symmetrical_brain_hubs ~= 1) = 0;
        [~,p_NSF,~,stats_NSF] = ttest(GT_NSF);
        t_NSF = stats_NSF.tstat;
        t_NSF(isnan(t_NSF)) = 0;
        p_fdr_NSF = zeros(size(p_NSF));
        p_fdr_NSF(symmetrical_brain_hubs == 1) = mafdr(p_NSF(symmetrical_brain_hubs == 1), 'BHFDR', true);
        t_fdr_NSF = zeros(1,length(t_NSF));
        t_fdr_NSF(p_NSF < 0.05) = t_NSF(p_NSF < 0.05);
        t_NSF([184,371]) = t_NSF([120,307]);
        t_NSF([120,307]) = 0;
        t_fdr_NSF([184,371]) = t_fdr_NSF([120,307]);
        t_fdr_NSF([120,307]) = 0;
        display(regions(abs(t_fdr_NSF) > 0));
        display(t_fdr_NSF(abs(t_fdr_NSF) > 0));
        
        SF_values2surf_cortex = parcel_to_surface(t_SF([1:180 188:367])','glasser_360_fsa5');
        SF_values2surf_subcortex = t_SF([181:187 368:end]);
        NSF_values2surf_cortex = parcel_to_surface(t_NSF([1:180 188:367])','glasser_360_fsa5');
        NSF_values2surf_subcortex = t_NSF([181:187 368:end]);
        figure(1),
        plot_cortical(SF_values2surf_cortex, ...
                          'surface_name','fsa5', ...
                          'cmap','RdBu_r',...
                          'color_range', [-max(abs([t_SF t_NSF])) max(abs([t_SF t_NSF]))]);
        print(gcf, ...
              [maindir(1:id(end) - 6),'/','Results','/','fs_prediction','/','figures','/',...
               sprintf('%s_%s_SF_brain_hubs_tstats_cortical_map.png',char(modality),char(GT_metric))],'-dpng','-r1000');
        figure(2),
        plot_subcortical(SF_values2surf_subcortex, ...
                           'ventricles', 'False', ...
                           'cmap','RdBu_r',...
                           'color_range', [-max(abs([t_SF t_NSF])) max(abs([t_SF t_NSF]))]);
        print(gcf, ...
              [maindir(1:id(end) - 6),'/','Results','/','fs_prediction','/','figures','/',...
               sprintf('%s_%s_SF_brain_hubs_tstats_subcortical_map.png',char(modality),char(GT_metric))],'-dpng','-r1000');
        figure(3),
        plot_cortical(NSF_values2surf_cortex, ...
                          'surface_name','fsa5', ...
                          'cmap','RdBu_r',...
                          'color_range', [-max(abs([t_SF t_NSF])) max(abs([t_SF t_NSF]))]);
        print(gcf, ...
              [maindir(1:id(end) - 6),'/','Results','/','fs_prediction','/','figures','/',...
               sprintf('%s_%s_NSF_brain_hubs_tstats_cortical_map.png',char(modality),char(GT_metric))],'-dpng','-r1000');
        figure(4),
        plot_subcortical(NSF_values2surf_subcortex, ...
                           'ventricles', 'False', ...
                           'cmap','RdBu_r',...
                           'color_range', [-max(abs([t_SF t_NSF])) max(abs([t_SF t_NSF]))]);
        print(gcf, ...
              [maindir(1:id(end) - 6),'/','Results','/','fs_prediction','/','figures','/',...
               sprintf('%s_%s_NSF_brain_hubs_tstats_subcortical_map.png',char(modality),char(GT_metric))],'-dpng','-r1000');
    end
end

%% ========================================================================
% NSF-SF comparison
results = table(cell(0,1),cell(0,1),cell(0,1),zeros(0,1),zeros(0,1),zeros(0,1),zeros(0,1),zeros(0,1),zeros(0,1),... 
          'VariableNames',{'modality','GT_metric','ROI','estimate','SE','t_stats','p_values','p_values_fdr','brain_hubs'});
for modality = {'SC','FC'}
    if strcmp(modality,'SC') == 1
        folder = 'MRtrix';
    elseif strcmp(modality,'FC') == 1  
        folder = ['TS/' atlas];
    end
    datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Patients','/derivatives','/',folder];
    T = readtable([datadir,'/','GT_all_' atlas '_prediction_mean_MI_H_atypical_ipsicontra.csv']);
    T.TypeOfSurgery = double(strcmp(T.TypeOfSurgery,'Res'));
    regions = T.Properties.VariableNames(27:end);
    T = table2cell(T(strcmp(T.Site,'Emory'),:));
    for GT_metric = {'BC','PC'}
        GT_data = T(strcmp(T(:,26),GT_metric),[2,27:end]);
        GT = cell2mat(GT_data(:,2:end));
        Status = categorical(GT_data(:,1));
        Status = reordercats(Status,{'SF','NSF'});
        Age_onset = cell2mat(T(strcmp(T(:,26),GT_metric),9));
        Epilepsy_duration = cell2mat(T(strcmp(T(:,26),GT_metric),10));
        Postsurgical_duration = cell2mat(T(strcmp(T(:,26),GT_metric),11));
        Lesion = cell2mat(T(strcmp(T(:,26),GT_metric),12));
        Status_spike = cell2mat(T(strcmp(T(:,26),GT_metric),18));
        Type_Surgery = cell2mat(T(strcmp(T(:,26),GT_metric),5));
        GM_volume = cell2mat(T(strcmp(T(:,26),GT_metric),24));
        WM_volume = cell2mat(T(strcmp(T(:,26),GT_metric),25));
        GT_SF = GT(strcmp(GT_data(:,1),'SF'),:);
        GT_NSF = GT(strcmp(GT_data(:,1),'NSF'),:);
        t_stats = zeros(length(regions), 1);
        t_stats_fpr = zeros(length(regions), 1);
        t_stats_fdr = zeros(length(regions), 1);
        p_values = zeros(length(regions), 1);

        for i = 1:length(regions)
            y = GT(:,i); 
            tbl = table(y,Status,Age_onset,Epilepsy_duration,Postsurgical_duration,Lesion,Status_spike,Type_Surgery,GM_volume,WM_volume,...
                        'VariableNames',{'y','Status','Age_onset','Epilepsy_duration','Postsurgical_duration',...
                        'Lesion','Status_spike','Type_Surgery','GM_volume','WM_volume'});
            lm = fitlm(tbl, 'y ~ 1 + Status + Age_onset + Epilepsy_duration + Lesion + Status_spike + GM_volume + WM_volume');
            coeffs = lm.Coefficients;
            b_values(i) = coeffs{'Status_NSF','Estimate'};
            se_values(i) = coeffs{'Status_NSF','SE'};
            t_stats(i) = coeffs{'Status_NSF','tStat'};
            p_values(i) = coeffs{'Status_NSF','pValue'};
            if p_values(i) < 0.05
               t_stats_fpr(i) = t_stats(i);
            end
            row = {modality,GT_metric,regions(i),b_values(i),se_values(i),t_stats(i),p_values(i),NaN,NaN};
            results = [results; cell2table(row,'VariableNames',results.Properties.VariableNames)];
        end
        % y_mean = mean(GT,2);
        % tbl_mean = table(y_mean, Status, Age_onset, Epilepsy_duration,Postsurgical_duration, Lesion, Status_spike, Type_Surgery, GM_volume, WM_volume, ...
        %     'VariableNames', {'y_mean','Status','Age_onset','Epilepsy_duration','Postsurgical_duration',...
        %                 'Lesion','Status_spike','Type_Surgery','GM_volume','WM_volume'} );
        % lm_mean = fitlm(tbl_mean,'y_mean ~ 1 + Status + Age_onset + Epilepsy_duration + Lesion + GM_volume + WM_volume');
        % coeffs_mean = lm_mean.Coefficients;
        % b_mean = coeffs_mean{'Status_NSF','Estimate'};
        % se_mean = coeffs_mean{'Status_NSF','SE'};
        % t_mean = coeffs_mean{'Status_NSF','tStat'};
        % p_mean  = coeffs_mean{'Status_NSF','pValue'};
        % row = {modality,GT_metric,'Whole_brain',b_mean,se_mean,t_mean,p_mean,NaN};
        % results = [results; cell2table(row,'VariableNames',results.Properties.VariableNames)];
        % p_values = [p_values;p_mean];
        % 
        % sig_idx = t_stats_fpr ~= 0;    
        % y_mean_sig = mean(GT(:,sig_idx),2); 
        % tbl_mean_sig = table(y_mean_sig, Status, Age_onset, Epilepsy_duration, ...
        %                   Postsurgical_duration, Lesion, Status_spike, Type_Surgery, GM_volume, WM_volume, 'VariableNames', ...
        %                   {'y_mean_sig','Status','Age_onset','Epilepsy_duration','Postsurgical_duration',...
        %                   'Lesion','Status_spike','Type_Surgery','GM_volume','WM_volume'});
        % lm_mean_sig = fitlm(tbl_mean_sig,'y_mean_sig ~ 1 + Status + Age_onset + Epilepsy_duration + Lesion + GM_volume + WM_volume');
        % coeffs_mean_sig = lm_mean_sig.Coefficients;
        % b_mean_sig  = coeffs_mean_sig{'Status_NSF','Estimate'};
        % se_mean_sig = coeffs_mean_sig{'Status_NSF','SE'};
        % t_mean_sig  = coeffs_mean_sig{'Status_NSF','tStat'};
        % p_mean_sig  = coeffs_mean_sig{'Status_NSF','pValue'};
        % row = {modality, GT_metric,'Whole_brain_sig',b_mean_sig, se_mean_sig, t_mean_sig, p_mean_sig, NaN};
        % results = [results; cell2table(row,'VariableNames', results.Properties.VariableNames)];
        % p_values = [p_values;p_mean_sig];
        
        p_fdr = mafdr(p_values,'BHFDR',true);
        t_stats_fdr(p_fdr < 0.05) = t_stats(p_fdr < 0.05);
        idx = strcmp(results.modality,modality) & strcmp(results.GT_metric,GT_metric);
        results.p_values_fdr(find(idx)) = p_fdr;
        t_stats([184,371]) = t_stats([120,307]);
        t_stats([120,307]) = 0;
        t_stats_fpr([184,371]) = t_stats_fpr([120,307]);
        t_stats_fpr([120,307]) = 0;
        if strcmp(GT_metric,'BC') == 1
            load([maindir(1:id(end) - 15),'/cMRI_BIDS','/code','/fs_ahubness','/',sprintf('%s_BC_brain_hubs.mat',char(modality))]);
            BC_final_hubs(BC_final_hubs < 2) = 0; BC_final_hubs(BC_final_hubs >= 2) = 1;
            brain_hubs = BC_final_hubs;
        elseif strcmp(GT_metric,'EC') == 1
            load([maindir(1:id(end) - 15),'/cMRI_BIDS','/code','/fs_ahubness','/',sprintf('%s_EC_brain_hubs.mat',char(modality))]);
            EC_final_hubs(EC_final_hubs < 2) = 0; EC_final_hubs(EC_final_hubs >= 2) = 1;
            brain_hubs = EC_final_hubs;
        elseif strcmp(GT_metric,'PC') == 1
            load([maindir(1:id(end) - 15),'/cMRI_BIDS','/code','/fs_ahubness','/',sprintf('%s_PC_brain_hubs.mat',char(modality))]);
            PC_final_hubs(PC_final_hubs < 2) = 0; PC_final_hubs(PC_final_hubs >= 2) = 1;
            brain_hubs = PC_final_hubs;
        end
        symmetrical_brain_hubs = zeros(size(brain_hubs));
        symmetrical_brain_hubs(1:180) = brain_hubs(1:180) & brain_hubs(181:360); 
        symmetrical_brain_hubs(181:360) = brain_hubs(1:180) & brain_hubs(181:360); 
        symmetrical_brain_hubs(361:367) = brain_hubs(361:367) & brain_hubs(368:end); 
        symmetrical_brain_hubs(368:end) = brain_hubs(361:367) & brain_hubs(368:end); 
        symmetrical_brain_hubs([364,371]) = symmetrical_brain_hubs([120,300]);
        symmetrical_brain_hubs([120,300]) = 0;
        symmetrical_brain_hubs = [symmetrical_brain_hubs(1:180) symmetrical_brain_hubs(361:367) symmetrical_brain_hubs(181:360) symmetrical_brain_hubs(368:end)];
        results.brain_hubs(find(idx)) = symmetrical_brain_hubs';
        t_stats_fpr(symmetrical_brain_hubs ~= 1) = 0;
        t_stats(symmetrical_brain_hubs ~= 1) = 0;

        t_stats2surf_cortex = parcel_to_surface(t_stats([1:180 188:367])','glasser_360_fsa5');
        t_stats2surf_subcortex = t_stats([181:187 368:end]);
        t_stats_fdr2surf_cortex = parcel_to_surface(t_stats_fdr([1:180 188:367])','glasser_360_fsa5');
        t_stats_fdr2surf_subcortex = t_stats_fdr([181:187 368:end]);
        t_stats_fpr2surf_cortex = parcel_to_surface(t_stats_fpr([1:180 188:367])','glasser_360_fsa5');
        t_stats_fpr2surf_subcortex = t_stats_fpr([181:187 368:end]);
        figure(1),
        plot_cortical(t_stats2surf_cortex, ...
                      'surface_name','fsa5', ...
                      'cmap','RdBu_r', ...
                      'color_range',[-max(t_stats),max(t_stats)]);
        print(gcf, ...
              [maindir(1:id(end) - 6),'/','Results','/','fs_prediction','/','figures','/',...
               sprintf('%s_%s_NSF-SF_brain_hubs_tstats_cortical_map.png',char(modality),char(GT_metric))],'-dpng','-r1000')
        figure(2),
        plot_subcortical(t_stats2surf_subcortex, ...
                         'ventricles', 'False', ...
                         'cmap', 'RdBu_r', ...
                         'color_range',[-max(t_stats),max(t_stats)]);
        print(gcf, ...
              [maindir(1:id(end) - 6),'/','Results','/','fs_prediction','/','figures','/',...
               sprintf('%s_%s_NSF-SF_brain_hubs_tstats_subcortical_map.png',char(modality),char(GT_metric))],'-dpng','-r1000')
        clear p_values t_stats t_stats_fpr t_stats_fdr
    end
end

%% ========================================================================
% plotting FI and t-values
FI = table2cell(readtable([maindir(1:id(end) - 6),'/Results','/fs_prediction','/tables','/FI_PC_FC-SC_HCP_MMP1_DK_all.csv']));
FI(:,[1,5,6,8]) = [];
for modality = {'SC','FC'}
    if strcmp(modality,'SC') == 1
        folder = 'MRtrix';
    elseif strcmp(modality,'FC') == 1  
        folder = ['TS/' atlas];
    end
    datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Patients','/derivatives','/',folder];
    T = readtable([datadir,'/','GT_all_' atlas '_prediction_mean_MI_H_atypical_ipsicontra.csv']);
    T.TypeOfSurgery = double(strcmp(T.TypeOfSurgery,'Res'));
    regions = T.Properties.VariableNames(27:end);
    T = table2cell(T(~ strcmp(T.Site,'Emory'),:));
    GT_data = T(strcmp(T(:,26),'PC'),[2,27:end]);
    GT = cell2mat(GT_data(:,2:end));
    Status = categorical(GT_data(:,1));
    Status = reordercats(Status,{'SF','NSF'});
    Age_onset = cell2mat(T(strcmp(T(:,26),'PC'),9));
    Epilepsy_duration = cell2mat(T(strcmp(T(:,26),'PC'),10));
    Postsurgical_duration = cell2mat(T(strcmp(T(:,26),'PC'),11));
    Lesion = cell2mat(T(strcmp(T(:,26),'PC'),12));
    Status_spike = cell2mat(T(strcmp(T(:,26),'PC'),18));
    Type_Surgery = cell2mat(T(strcmp(T(:,26),'PC'),5));
    GM_volume = cell2mat(T(strcmp(T(:,26),'PC'),24));
    WM_volume = cell2mat(T(strcmp(T(:,26),'PC'),25));
    t_stats = zeros(length(regions), 1);

    for i = 1:length(regions)
        y = GT(:,i); 
        tbl = table(y,Status,Age_onset,Epilepsy_duration,Postsurgical_duration,Lesion,Status_spike,Type_Surgery,GM_volume,WM_volume,...
                    'VariableNames',{'y','Status','Age_onset','Epilepsy_duration','Postsurgical_duration',...
                    'Lesion','Status_spike','Type_Surgery','GM_volume','WM_volume'});
        lm = fitlm(tbl, 'y ~ 1 + Status + Age_onset + Epilepsy_duration + Lesion + Status_spike + GM_volume + WM_volume');
        coeffs = lm.Coefficients;
        b_values(i) = coeffs{'Status_NSF','Estimate'};
        se_values(i) = coeffs{'Status_NSF','SE'};
        t_stats(i) = coeffs{'Status_NSF','tStat'};
        p_values(i) = coeffs{'Status_NSF','pValue'};
    end
    t_stats([184,371]) = t_stats([120,307]);
    t_stats([120,307]) = 0;
    p_values([184,371]) = p_values([120,307]);
    p_values([120,307]) = 0;
    load([maindir(1:id(end) - 15),'/cMRI_BIDS','/code','/fs_ahubness','/',sprintf('%s_PC_brain_hubs.mat',char(modality))]);
    PC_final_hubs(PC_final_hubs < 2) = 0; PC_final_hubs(PC_final_hubs >= 2) = 1;
    brain_hubs = PC_final_hubs;
    symmetrical_brain_hubs = zeros(size(brain_hubs));
    symmetrical_brain_hubs(1:180) = brain_hubs(1:180) & brain_hubs(181:360); 
    symmetrical_brain_hubs(181:360) = brain_hubs(1:180) & brain_hubs(181:360); 
    symmetrical_brain_hubs(361:367) = brain_hubs(361:367) & brain_hubs(368:end); 
    symmetrical_brain_hubs(368:end) = brain_hubs(361:367) & brain_hubs(368:end); 
    symmetrical_brain_hubs([364,371]) = symmetrical_brain_hubs([120,300]);
    symmetrical_brain_hubs([120,300]) = 0;
    symmetrical_brain_hubs = [symmetrical_brain_hubs(1:180) symmetrical_brain_hubs(361:367) symmetrical_brain_hubs(181:360) symmetrical_brain_hubs(368:end)];
    Ci_modality = Ci;
    Ci_modality([364,371]) = Ci_modality([120,300]);
    Ci_modality([120,300]) = 0;
    Ci_modality = [Ci_modality(1:180); Ci_modality(361:367); Ci_modality(181:360); Ci_modality(368:end)];
    Ci_modality(symmetrical_brain_hubs ~= 1) = 0;
    t_stats(symmetrical_brain_hubs ~= 1) = 0;
    p_values(symmetrical_brain_hubs ~= 1) = 0;

    FI_modality = FI(contains(FI(:,1),[char(modality) '_']),:);
    FI_modality(:,[3,4]) = [];
    FI_modality(:,1) = erase(FI_modality(:,1),[char(modality) '_']);
    networks = str2double(erase((FI_modality(:,1)), 'group_'));
    FI_modality = [networks cell2mat(FI_modality(:,2))];
    fi_values = zeros(length(regions),1);
    for network = networks'
        fi_values(find(Ci_modality == network)) = FI_modality(FI_modality(:,1) == network,2);
    end

    t_stats2surf_cortex = parcel_to_surface(t_stats([1:180 188:367])','glasser_360_fsa5');
    t_stats2surf_subcortex = t_stats([181:187 368:end]);
    fi_values2surf_cortex = parcel_to_surface(fi_values([1:180 188:367])','glasser_360_fsa5');
    fi_values2surf_subcortex = fi_values([181:187 368:end]);
    figure(1),
    plot_cortical(t_stats2surf_cortex, ...
                  'surface_name','fsa5', ...
                  'cmap','RdBu_r', ...
                  'color_range',[-max(t_stats),max(t_stats)]);
    print(gcf, ...
          [maindir(1:id(end) - 6),'/','Results','/','fs_prediction','/','figures','/',...
           sprintf('%s_PC_FI_brain_hubs_tstats_cortical_map.png',char(modality))],'-dpng','-r1000')
    figure(2),
    plot_subcortical(t_stats2surf_subcortex, ...
                     'ventricles', 'False', ...
                     'cmap', 'RdBu_r', ...
                     'color_range',[-max(t_stats),max(t_stats)]);
    print(gcf, ...
          [maindir(1:id(end) - 6),'/','Results','/','fs_prediction','/','figures','/',...
           sprintf('%s_PC_FI_brain_hubs_tstats_subcortical_map.png',char(modality))],'-dpng','-r1000')
    figure(3),
    plot_cortical(fi_values2surf_cortex, ...
                  'surface_name','fsa5', ...
                  'cmap','RdBu_r', ...
                  'color_range',[-max(cell2mat(FI(:,2))),max(cell2mat(FI(:,2)))]);
    print(gcf, ...
          [maindir(1:id(end) - 6),'/','Results','/','fs_prediction','/','figures','/',...
           sprintf('%s_PC_FI_brain_hubs_fi_values_cortical_map.png',char(modality))],'-dpng','-r1000')
    figure(4),
    plot_subcortical(fi_values2surf_subcortex, ...
                     'ventricles', 'False', ...
                     'cmap', 'RdBu_r', ...
                     'color_range',[-max(cell2mat(FI(:,2))),max(cell2mat(FI(:,2)))]);
    print(gcf, ...
          [maindir(1:id(end) - 6),'/','Results','/','fs_prediction','/','figures','/',...
           sprintf('%s_PC_FI_brain_hubs_fi_values_subcortical_map.png',char(modality))],'-dpng','-r1000')
    clear t_stats fi_values Ci_modality
end

%% ========================================================================
% sensitivity analyses
% patients
GT_datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Patients','/derivatives','/','TS/' atlas];
GT_T = table2cell(readtable([GT_datadir,'/','GT_all_' atlas '_prediction_mean_MI_H.csv']));
GT_wnc_T = table2cell(readtable([GT_datadir,'/','GT_all_' atlas '_prediction_mean_wnc_MI_H.csv']));
GT_wopt_T = table2cell(readtable([GT_datadir,'/','GT_all_' atlas '_prediction_mean_wopt_MI_H.csv']));
GT_data = cell2mat(GT_T(strcmp(GT_T(:,26),'PC'),27:end));
GT_data_wnc = cell2mat(GT_wnc_T(strcmp(GT_wnc_T(:,26),'PC'),27:end));
GT_data_wopt = cell2mat(GT_wopt_T(strcmp(GT_wopt_T(:,26),'PC'),27:end));

PC_mean = mean(GT_data([1:180 188:367]),1);
PC_mean_wnc = mean(GT_data_wnc([1:180 188:367]),1);
PC_mean_wopt = mean(GT_data_wopt([1:180 188:367]),1);
[p_wnc, d_wnc] = spin_test(PC_mean,PC_mean_wnc,...
                           'surface_name','fsa5',...
                           'parcellation_name','glasser_360',...
                           'n_rot',5000,...
                           'type','spearman');
r_wnc = corr(PC_mean',PC_mean_wnc','Type','Spearman');

[p_wopt, d_wopt] = spin_test(PC_mean,PC_mean_wopt,...
                           'surface_name','fsa5',...
                           'parcellation_name','glasser_360',...
                           'n_rot',5000,...
                           'type','spearman');
r_wopt = corr(PC_mean',PC_mean_wopt','Type','Spearman');
