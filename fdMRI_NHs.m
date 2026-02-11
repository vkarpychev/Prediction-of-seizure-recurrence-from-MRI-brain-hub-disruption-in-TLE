%% ========================================================================
% folders
clear
atlas = 'HCP_MMP1_DK';
maindir = pwd;
id = strfind(maindir,'/');
maskdir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/masks','/',atlas];
datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Controls'];
addpath(genpath([maindir(1:id(end) - 6),'/code','/libraries/']));
dcontrol_list = table2cell(readtable('dCL_ahubness.txt'));
fcontrol_list = table2cell(readtable('fCL_ahubness.txt'));

%% ========================================================================
for modality = {'SC','FC'}
    if strcmp(modality,'SC') == 1
        folder = 'MRtrix';
    elseif strcmp(modality,'FC') == 1  
        folder = ['TS/',atlas];
    end
    GT_datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Controls','/derivatives','/',folder];
    GT_T = readtable([GT_datadir,'/','GT_all_' atlas '_ahubness_mean_MI_H.csv']);
    for GT_metric = {'BC','PC'}
        GT_data = GT_T(strcmp(GT_T{:,7},GT_metric),8:end);
        typical_values = mean(table2array(GT_data));
        if strcmp(GT_metric,'BC') == 1
            load(sprintf('%s_BC_brain_hubs.mat',char(modality)));
            BC_final_hubs(BC_final_hubs < 2) = 0; BC_final_hubs(BC_final_hubs >= 2) = 1;
            brain_hubs = BC_final_hubs;
        elseif strcmp(GT_metric,'EC') == 1
            load(sprintf('%s_EC_brain_hubs.mat',char(modality)));
            EC_final_hubs(EC_final_hubs < 2) = 0; EC_final_hubs(EC_final_hubs >= 2) = 1;
            brain_hubs = EC_final_hubs;
        elseif strcmp(GT_metric,'PC') == 1
            load(sprintf('%s_PC_brain_hubs.mat',char(modality)));
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
        display(sum(symmetrical_brain_hubs));
        brain_hubs2surf_cortex = parcel_to_surface(symmetrical_brain_hubs(1:360)','glasser_360_fsa5');
        brain_hubs2surf_subcortex = symmetrical_brain_hubs(361:end);

        % figure(1),
        % plot_cortical(typical_values2surf_cortex, ...
        %               'surface_name','fsa5', ...
        %               'cmap','Reds',...
        %               'color_range', [min(typical_values) max(typical_values)]);
        % print(gcf, ...
        %       [maindir(1:id(end) - 6),'/','Results','/','fs_ahubness','/','figures','/',...
        %        sprintf('%s_%s_cortical_map_mean_values.png',char(modality),char(GT_metric))],...
        %        '-dpng','-r1000');
        % figure(2),
        % plot_subcortical(typical_values2surf_subcortex, ...
        %                  'ventricles', 'False', ...
        %                  'cmap','Reds',...
        %                  'color_range', [min(typical_values) max(typical_values)]);
        % print(gcf, ...
        %       [maindir(1:id(end) - 6),'/','Results','/','fs_ahubness','/','figures','/',...
        %        sprintf('%s_%s_subcortical_map_mean_values.png',char(modality),char(GT_metric))],...
        %        '-dpng','-r1000');
        figure(1),
        plot_cortical(brain_hubs2surf_cortex, ...
                      'surface_name','fsa5', ...
                      'cmap','Reds',...
                      'color_range', [min(brain_hubs) max(brain_hubs)]);
        print(gcf, ...
              [maindir(1:id(end) - 6),'/','Results','/','fs_ahubness','/','figures','/',...
               sprintf('%s_%s_brain_hubs_cortical_map_mean_values.png',char(modality),char(GT_metric))],...
               '-dpng','-r1000');
        figure(2),
        plot_subcortical(brain_hubs2surf_subcortex, ...
                         'ventricles', 'False', ...
                         'cmap','Reds',...
                         'color_range', [min(brain_hubs) max(brain_hubs)]);
        print(gcf, ...
              [maindir(1:id(end) - 6),'/','Results','/','fs_ahubness','/','figures','/',...
               sprintf('%s_%s_brain_hubs_subcortical_map_mean_values.png',char(modality),char(GT_metric))],...
               '-dpng','-r1000');
    end
end

%% ========================================================================
% controls
GT_datadir = [maindir(1:id(end) - 6),'/Data','/ImageDatabase_BIDS','/Controls','/derivatives','/','TS/' atlas];
GT_T = table2cell(readtable([GT_datadir,'/','GT_all_' atlas '_ahubness_mean_MI_H.csv']));
GT_wnc_T = table2cell(readtable([GT_datadir,'/','GT_all_' atlas '_ahubness_mean_wnc_MI_H.csv']));
GT_wopt_T = table2cell(readtable([GT_datadir,'/','GT_all_' atlas '_ahubness_mean_wopt_MI_H.csv']));
GT_data = cell2mat(GT_T(strcmp(GT_T(:,7),'PC'),27:end));
GT_data_wnc = cell2mat(GT_wnc_T(strcmp(GT_wnc_T(:,7),'PC'),27:end));
GT_data_wopt = cell2mat(GT_wopt_T(strcmp(GT_wopt_T(:,7),'PC'),27:end));

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