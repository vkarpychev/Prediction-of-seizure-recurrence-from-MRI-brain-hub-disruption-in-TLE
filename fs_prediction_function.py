#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 7 15:14:21 2025

@author: victor
"""

def run_in_batches(func, total_iterations, batch_size):
    
    from multiprocessing import Pool
    
    results = []
    for i in range(0, total_iterations, batch_size):
        batch_range = list(range(i, min(i + batch_size, total_iterations)))
        batch_args = [(j,) for j in batch_range] 
        print(f'Running batch: {batch_range}')
        with Pool(processes = batch_size) as pool:
            batch_results = pool.starmap(func, batch_args)
            results.extend(batch_results)
    return results

def npv_score(y_true, y_pred, threshold = 0.5):
    
    from sklearn.metrics import confusion_matrix
    
    y_pred = y_pred >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

def fs_feature_importance(test_sets, shap_values_sets, features, metric, modality, site):
    
    import numpy as np
    import shap
    import matplotlib.pyplot as plt
    
    if site == 0:
        shap_values_all = np.array(shap_values_sets[1])
        test_set_all = test_sets[1]
        for i in range(1, len(test_sets)):
            test_set_all = np.concatenate((test_set_all, test_sets[i]), axis = 0)
            shap_values_all = np.concatenate((shap_values_all, np.array(shap_values_sets[i])), axis = 0)
        fig, ax = plt.subplots()
        plt.title(f' Long-term outcome {modality}', fontsize = 14, fontweight = 'bold', pad = 5, loc = 'left')
        shap.summary_plot(shap_values_all, features = test_set_all, feature_names = np.array(features), max_display = len(features))
     #   fig.savefig(f'/media/rgong/MRI_BIDS/Results/fs_prediction/figures/FI/FI_{metric}_{modality}.png', dpi = 1000, bbox_inches = 'tight', pad_inches = 0.3) 
        return
    else:
        fig, ax = plt.subplots()
        plt.title(f' Long-term outcome {modality} {site}', fontsize = 14, fontweight = 'bold', pad = 5, loc = 'left')
        shap.summary_plot(shap_values_sets, features = test_sets, feature_names = np.array(features), max_display = 15)
        fig.savefig(f'/media/rgong/MRI_BIDS/Results/fs_prediction/figures/FI/FI_{metric}_{modality}_{site}.png', dpi = 1000, bbox_inches = 'tight', pad_inches = 0.3) 
        return

def compute_metrics(y_true, y_prob, thr = 0.5):
    
    import numpy as np
    from sklearn import metrics
    
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    brier = metrics.brier_score_loss(np.asarray(y_true).astype(int), y_prob)
    
    cm = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ppv = metrics.precision_score(y_true, y_pred, zero_division = 0)
    brier = metrics.brier_score_loss(y_true, y_prob)
    return {
            'roc_auc': metrics.roc_auc_score(y_true, y_prob),
            'accuracy': metrics.accuracy_score(y_true, y_pred),
            'ppv': ppv,
            'recall': metrics.recall_score(y_true, y_pred),
            'specificity': specificity,
            'npv': npv,
            'f1-score': metrics.f1_score(y_true, y_pred),
            'brier': brier
            }

def expand_metrics(report_list, model, modality):
    
    import pandas as pd
    df = pd.DataFrame(report_list)
    df['model'] = model
    df['modality'] = modality
    return df

def youden_index(y_true, y_prob):
    
    import numpy as np
    from sklearn import metrics

    fpr, tpr, thr = metrics.roc_curve(y_true, y_prob)   
    j = tpr - fpr                             
    return thr[np.argmax(j)]

def npv_score_ruleout(y_true, y_prob, target_npv = 0.90, grid = None):
    
    import numpy as np
    
    if grid is None:
        grid = np.linspace(0.01, 0.99, 199)
    npvs = np.array([npv_score(y_true, y_prob, t) for t in grid])
    idx = np.where(npvs >= target_npv)[0]
    if len(idx) == 0:

        return float(grid[np.argmax(npvs)])
    return float(grid[idx[-1]])

def net_benefit(y, p, pt):
    
    import numpy as np
    
    yhat = (p >= pt).astype(int)
    n = len(y)
    tp = np.sum((yhat == 1) & (y == 1))
    fp = np.sum((yhat == 1) & (y == 0))
    w = pt / (1 - pt)
    return (tp / n) - (fp / n) * w

def fs_net_benefit(nb_model_sets, nb_all_sets, nb_none_sets, thresholds_sets):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    nb_model = np.mean(nb_model_sets, axis = 0)
    nb_all = np.mean(nb_all_sets, axis = 0)
    nb_none = np.mean(nb_none_sets, axis = 0)
    thresholds = np.mean(thresholds_sets, axis = 0)
    
    plt.figure()
    plt.plot(thresholds, nb_model, label = 'Model')
    plt.plot(thresholds, nb_all, label = 'Exclude-all')
    plt.plot(thresholds, nb_none, label = 'Exclude-none')
    plt.xlabel('Threshold probability P(fail)')
    plt.ylabel('Net benefit')
    plt.legend()
    plt.show()

def fs_run_iteration_cv_models(i, df, metric, modality):
    
    import numpy as np
    import pandas as pd
    from sklearn import model_selection, feature_selection, compose, calibration
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as sklearn_Pipeline
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.dummy import DummyClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score    
    from feature_engine.outliers import Winsorizer
    import warnings
    import time
    import shap

    print(f'# iteration {i}')
    
    X, y = df.iloc[:, 2:], df.iloc[:, 0]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = i, stratify = y)
    
    # %% =============================================================================
    # Dummy
    clf = DummyClassifier(strategy = 'most_frequent')
    clf.fit(X_train, y_train)
    null_metrics = compute_metrics(y_test, clf.predict(X_test), 0.5)

    # %% ======================================================================
    # Gradient Boosting
    # %% ======================================================================
    scorers = {'roc_auc':'roc_auc','accuracy':'accuracy','ppv': make_scorer(precision_score, zero_division = 0),               
                'recall': make_scorer(recall_score, zero_division = 0),'specificity': make_scorer(recall_score, pos_label = 0, zero_division = 0),
                 'npv': make_scorer(npv_score),'f1-score': make_scorer(f1_score, zero_division = 0)}
    if modality != 'DC':
        preprocessing = compose.ColumnTransformer(
            transformers = [
                ('scale_covars', sklearn_Pipeline([('scaling', StandardScaler()),
                                                   ]), slice(0,3)),
                ('winsor_hubs',
                 sklearn_Pipeline([
                     ('outliers', Winsorizer(
                         capping_method = 'iqr',
                         tail = 'both',
                         fold = 2.5,
                         variables = X_train.columns[7:].to_list()
                         )),
                     ]),
                 slice(7,None)),
                ],
            remainder = 'drop')
    else:
         preprocessing = compose.ColumnTransformer(
             transformers = [
                 ('scale_covars', sklearn_Pipeline([('scaling', StandardScaler()),
                                                    ]), slice(0,3))                 ],
             remainder = 'drop')   
  
    # %% ======================================================================
    gb_clf = GradientBoostingClassifier(random_state = 0)
    gb_pipe = Pipeline([
             ('preprocess', preprocessing),
             ('mi_select', feature_selection.SelectKBest(score_func = feature_selection.mutual_info_classif)),
             ('smote', SMOTE(random_state = 0)),
             ('classifier', gb_clf),
             ])
    gb_params = {
                'mi_select__k': [5, 10, 15, 20,'all'],
                'classifier__n_estimators': [50, 100, 150, 200, 250, 300],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 4, 5, 7],
                'classifier__min_samples_split': [2, 5, 10, 15],
                'classifier__min_samples_leaf': [1, 2, 4, 6],
                'classifier__subsample': [0.6, 0.8, 1.0]
                 }
    start_time = time.time()
    gb_grid = model_selection.RandomizedSearchCV(gb_pipe, 
                                                 gb_params,
                                                 n_iter = 50,
                                                 cv = 4, 
                                                 scoring = scorers,
                                                 refit = 'roc_auc',
                                                 random_state = 0)
    warnings.filterwarnings('ignore', message = '.*BaseEstimator._validate_data.*', category = FutureWarning)
    gb_grid.fit(X_train, y_train) 
    cv_results = pd.DataFrame(gb_grid.cv_results_)
    best_idx = cv_results['mean_test_roc_auc'].idxmax()
    cv_metrics = {
                  'roc_auc': cv_results.loc[best_idx]['mean_test_roc_auc'],
                  'accuracy': cv_results.loc[best_idx]['mean_test_accuracy'],
                  'ppv': cv_results.loc[best_idx]['mean_test_ppv'],
                  'npv': cv_results.loc[best_idx]['mean_test_npv'],
                  'recall': cv_results.loc[best_idx]['mean_test_recall'],
                  'specificity': cv_results.loc[best_idx]['mean_test_specificity'],
                  'f1-score': cv_results.loc[best_idx]['mean_test_f1-score']
                   }    
    gb_cv = cv_metrics
    gb_best = gb_grid.best_estimator_
    calibrated_clf = calibration.CalibratedClassifierCV(estimator = gb_best,
                                                        method = 'sigmoid',          
                                                        cv = 4)
    calibrated_clf.fit(X_train, y_train)
    threshold = youden_index(y_train, calibrated_clf.predict_proba(X_train)[:, 1])
    gb_metrics = compute_metrics(y_test, calibrated_clf.predict_proba(X_test)[:, 1], threshold)   
    gb_test = gb_metrics
    auc_ci, npv_ci = bootstrap_ci(y_test, calibrated_clf.predict_proba(X_test)[:, 1], threshold)
        
    print('GB classifier')
    print('--- %s minutes ---' % ((time.time() - start_time) / 60))
    del start_time
          
    # %% ======================================================================
    # Feature importance
    preprocess = gb_best.named_steps['preprocess']
    selector = gb_best.named_steps['mi_select']  
    model = gb_best.named_steps['classifier']

    X_train_pre = preprocess.transform(X_train)
    X_test_pre  = preprocess.transform(X_test)
    mask = selector.get_support()

    X_train_sel = X_train_pre[:, mask]
    X_test_sel  = X_test_pre[:, mask]

    explainer = shap.TreeExplainer(model,
                                   X_train_sel,
                                   feature_perturbation = 'interventional'
                                   )
    shap_sel = explainer.shap_values(X_test_sel, check_additivity = False)
    odds_ratio_sel = model.feature_importances_.ravel()

    n_test = X_test.shape[0]
    p = X_test.shape[1]

    shap_values = np.zeros((n_test, p))
    odds_ratio_full = np.zeros(p)
    X_test_values = np.zeros((n_test, p))
    if modality != 'DC':
        sel_cols_pre = np.concatenate([np.arange(0, 3), np.arange(7, p)])
    else:
        sel_cols_pre = np.arange(0, 3)
    sel_cols_final = sel_cols_pre[mask]

    shap_values[:, sel_cols_final] = shap_sel
    odds_ratio_full[sel_cols_final] = odds_ratio_sel
    X_test_values[:, sel_cols_final] = X_test_sel

    return {'i': i,
            'gb_cv': gb_cv,
            'metrics': gb_test,
            'null_metrics': null_metrics,
            'shap': shap_values,
            'odds_ratio': odds_ratio_full,
            'test_set': X_test_values,
             'auc_ci': auc_ci,
             'npv_ci': npv_ci
            }

def fs_run_iteration_best_models(i, df, metric, modality):
    
    import numpy as np
    import pandas as pd
    from sklearn import model_selection, feature_selection, compose, calibration
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as sklearn_Pipeline
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.dummy import DummyClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score    
    from feature_engine.outliers import Winsorizer
    import warnings
    import time
    import shap

    print(f'# iteration {i}')
    
    X_train = df.loc[df['Site'] == 'Emory'].iloc[:,2:]
    y_train = df.loc[df['Site'] == 'Emory'].iloc[:,0]
    X_test = df.loc[df['Site'] != 'Emory'].iloc[:,2:]
    y_test = df.loc[df['Site'] != 'Emory'].iloc[:,0]
    
    # %% =============================================================================
    # Dummy
    clf = DummyClassifier(strategy = 'most_frequent')
    clf.fit(X_train, y_train)
    null_metrics = compute_metrics(y_test, clf.predict(X_test), 0.5)

    # %% ======================================================================
    # Gradient Boosting
    # %% ======================================================================
    scorers = {'roc_auc':'roc_auc','accuracy':'accuracy','ppv': make_scorer(precision_score, zero_division = 0),               
                'recall': make_scorer(recall_score, zero_division = 0),'specificity': make_scorer(recall_score, pos_label = 0, zero_division = 0),
                 'npv': make_scorer(npv_score),'f1-score': make_scorer(f1_score, zero_division = 0)}
    if modality != 'DC':
        preprocessing = compose.ColumnTransformer(
            transformers = [
                ('scale_covars', sklearn_Pipeline([('scaling', StandardScaler()),
                                                   ]), slice(0,3)),
                ('winsor_hubs',
                 sklearn_Pipeline([
                     ('outliers', Winsorizer(
                         capping_method = 'iqr',
                         tail = 'both',
                         fold = 2.5,
                         variables = X_train.columns[7:].to_list()
                         )),
                     ]),
                 slice(7,None)),
                ],
            remainder = 'drop')
    else:
         preprocessing = compose.ColumnTransformer(
             transformers = [
                 ('scale_covars', sklearn_Pipeline([('scaling', StandardScaler()),
                                                    ]), slice(0,3))                 ],
             remainder = 'drop')   
  
    # %% ======================================================================
    gb_clf = GradientBoostingClassifier(random_state = 0)
    gb_pipe = Pipeline([
             ('preprocess', preprocessing),
             ('mi_select', feature_selection.SelectKBest(score_func = feature_selection.mutual_info_classif)),
             ('smote', SMOTE(random_state = 0)),
             ('classifier', gb_clf),
             ])
    gb_params = {
                'mi_select__k': [5, 10, 15, 20,'all'],
                'classifier__n_estimators': [50, 100, 150, 200, 250, 300],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 4, 5, 7],
                'classifier__min_samples_split': [2, 5, 10, 15],
                'classifier__min_samples_leaf': [1, 2, 4, 6],
                'classifier__subsample': [0.6, 0.8, 1.0]
                 }
    start_time = time.time()
    cv = model_selection.StratifiedKFold(n_splits = 4, 
                                         shuffle = True, 
                                         random_state = i)
    gb_grid = model_selection.RandomizedSearchCV(gb_pipe, 
                                                 gb_params,
                                                 n_iter = 50,
                                                 cv = cv, 
                                                 scoring = scorers,
                                                 refit = 'roc_auc',
                                                 random_state = 0)
    warnings.filterwarnings('ignore', message = '.*BaseEstimator._validate_data.*', category = FutureWarning)
    gb_grid.fit(X_train, y_train) 
    cv_results = pd.DataFrame(gb_grid.cv_results_)
    best_idx = cv_results['mean_test_roc_auc'].idxmax()
    cv_metrics = {
                  'roc_auc': cv_results.loc[best_idx]['mean_test_roc_auc'],
                  'accuracy': cv_results.loc[best_idx]['mean_test_accuracy'],
                  'ppv': cv_results.loc[best_idx]['mean_test_ppv'],
                  'npv': cv_results.loc[best_idx]['mean_test_npv'],
                  'recall': cv_results.loc[best_idx]['mean_test_recall'],
                  'specificity': cv_results.loc[best_idx]['mean_test_specificity'],
                  'f1-score': cv_results.loc[best_idx]['mean_test_f1-score']
                   }    
    gb_cv = cv_metrics
    gb_best = gb_grid.best_estimator_
    calibrated_clf = calibration.CalibratedClassifierCV(estimator = gb_best,
                                                        method = 'sigmoid',          
                                                        cv = cv)
    calibrated_clf.fit(X_train, y_train)
    threshold = youden_index(y_train, calibrated_clf.predict_proba(X_train)[:, 1])
    gb_metrics = compute_metrics(y_test, calibrated_clf.predict_proba(X_test)[:, 1], threshold)   
    gb_test = gb_metrics
    auc_ci, npv_ci = bootstrap_ci(y_test, calibrated_clf.predict_proba(X_test)[:, 1], threshold)                           
        
    print('GB classifier')
    print('--- %s minutes ---' % ((time.time() - start_time) / 60))
    del start_time
          
    # %% ======================================================================
    # Feature importance
    preprocess = gb_best.named_steps['preprocess']
    selector = gb_best.named_steps['mi_select']  
    model = gb_best.named_steps['classifier']

    X_train_pre = preprocess.transform(X_train)
    X_test_pre  = preprocess.transform(X_test)
    mask = selector.get_support()

    X_train_sel = X_train_pre[:, mask]
    X_test_sel  = X_test_pre[:, mask]

    explainer = shap.TreeExplainer(model,
                                   X_train_sel,
                                   feature_perturbation = 'interventional'
                                   )
    shap_sel = explainer.shap_values(X_test_sel, check_additivity = False)
    odds_ratio_sel = model.feature_importances_.ravel()

    n_test = X_test.shape[0]
    p = X_test.shape[1]

    shap_values = np.zeros((n_test, p))
    odds_ratio_full = np.zeros(p)
    X_test_values = np.zeros((n_test, p))
    if modality != 'DC':
        sel_cols_pre = np.concatenate([np.arange(0, 3), np.arange(7, p)])
    else:
        sel_cols_pre = np.arange(0, 3)
    sel_cols_final = sel_cols_pre[mask]

    shap_values[:, sel_cols_final] = shap_sel
    odds_ratio_full[sel_cols_final] = odds_ratio_sel
    X_test_values[:, sel_cols_final] = X_test_sel

    return {'i': i,
            'gb_cv': gb_cv,
            'metrics': gb_test,
            'null_metrics': null_metrics,
            'shap': shap_values,
            'odds_ratio': odds_ratio_full,
            'test_set': X_test_values,
             'auc_ci': auc_ci,
             'npv_ci': npv_ci
            }
    
def fs_classifier_cv_parallel(df, metric, modality):
   
    from functools import partial
    import pandas as pd

    report_null = []
    report_grid_model = []  
    report_cv_model = []
    test_sets = []
    shap_values_sets = []
    odds_ratio_sets = []
    auc_ci_sets = []
    npv_ci_sets = []
    results_all = []
    
    worker = partial(fs_run_iteration_best_models, df = df, metric = metric, modality = modality)
  #  worker = partial(fs_run_iteration_cv_models, df = df, metric = metric, modality = modality)
    results = run_in_batches(worker, total_iterations = 100, batch_size = 5)

    report_null.extend([res['null_metrics'] for res in results])
    report_grid_model.extend([res['metrics'] for res in results])
    report_cv_model.extend([res['gb_cv'] for res in results])
    shap_values_sets.extend([res['shap'] for res in results])
    odds_ratio_sets.extend([res['odds_ratio'] for res in results])
    test_sets.extend([res['test_set'] for res in results])
    auc_ci_sets.extend([res['auc_ci'] for res in results])
    npv_ci_sets.extend([res['npv_ci'] for res in results])

    results_all = pd.concat([
         expand_metrics(report_null,'null model',modality),
         expand_metrics(report_grid_model, 'model', modality),
         expand_metrics(report_cv_model, 'cv', modality)
    ], ignore_index = True)
    
    features = df.columns[2:]
    fs_feature_importance(test_sets, shap_values_sets, features, metric, modality, site = 0)
    
    return results_all, shap_values_sets, odds_ratio_sets, auc_ci_sets, npv_ci_sets, features
