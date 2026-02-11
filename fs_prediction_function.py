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

def bootstrap_ci(y_true, y_proba, threshold, B = 1000, seed = None):
    
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_auc, boot_npv = [], []

    for _ in range(B):
        idx = rng.integers(0, n, size = n)
        if len(np.unique(y_true[idx])) >= 2:
            boot_auc.append(roc_auc_score(y_true[idx], y_proba[idx]))

        y_pred_b = (y_proba[idx] >= threshold).astype(int)
        denom = np.sum(y_pred_b == 0)
        if denom > 0:
            boot_npv.append(np.sum((y_pred_b == 0) & (y_true[idx] == 0)) / denom)

    auc_ci = np.percentile(boot_auc, [2.5, 97.5]) if len(boot_auc) else (np.nan, np.nan)
    npv_ci = np.percentile(boot_npv, [2.5, 97.5]) if len(boot_npv) else (np.nan, np.nan)

    return auc_ci, npv_ci

def fs_run_iteration_all_models(i, X_selected, y):
    
    import numpy as np
    import pandas as pd
    from sklearn import model_selection, feature_selection, compose
    from sklearn.preprocessing import StandardScaler
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression, LassoCV
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline as sklearn_Pipeline
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score    
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    import warnings
    import time

    X_train = X_selected['train']
    X_test = X_selected['test']
    y_train = np.ravel(y['train'])
    y_test = np.ravel(y['test'])

    # %% ======================================================================
    # Dummy
    clf = DummyClassifier(strategy = 'most_frequent')
    clf.fit(X_train, y_train)
    null_metrics = compute_metrics(y_test, clf.predict(X_test))

    # %% ======================================================================
    scorers = {'roc_auc':'roc_auc','accuracy':'accuracy','ppv': make_scorer(precision_score, zero_division = 0),               
               'recall': make_scorer(recall_score, zero_division = 0),'specificity': make_scorer(recall_score, pos_label = 0, zero_division = 0),
                'npv': make_scorer(npv_score),'f1-score': make_scorer(f1_score, zero_division = 0)}
    preprocessing = compose.ColumnTransformer(
    transformers = [
        ('all', sklearn_Pipeline([
            ('scaler', StandardScaler()),
            ('f_test', feature_selection.SelectFpr(
                score_func = feature_selection.f_classif,
                alpha = 0.05
            )),
        ]), slice(0, X_train.shape[1]))
        ], 
    remainder = 'drop')
    
    # %% ======================================================================
    rf_cv = []
    rf_test = []
    gb_cv = []
    gb_test = []
    xgb_cv = []
    xgb_test = []
    svm_cv = []
    svm_test = []
    lr_cv = []
    lr_test = []
    knn_cv = []
    knn_test = []
    lgbm_cv = []
    lgbm_test = []
    StClf_cv = []
    StClf_test = []
    
    for i in range(0,20):
          print(f'# iteration {i}')
          cv = model_selection.StratifiedKFold(n_splits = 4, 
                                               shuffle = True, 
                                               random_state = i)
          # %% ================================================================
          # RF
          start_time = time.time()
          rf_clf = RandomForestClassifier(random_state = 0)
          rf_pipe = Pipeline([
              ('preprocess', preprocessing),
              ('smote', SMOTE(random_state = 0)),
              ('feature_selection', feature_selection.SelectFromModel(LassoCV(max_iter = 10000,
                                                                              tol = 1e-3,
                                                                              cv = 4, 
                                                                              random_state = 0))), 
              ('classifier', rf_clf)
              ], verbose = False)
          rf_params = {
                 'feature_selection__threshold': ['median', 'mean', 0.0],
                 'feature_selection__estimator__alphas': [np.logspace(-2, 1, 10)],
                 'classifier__n_estimators': [50, 100, 150, 200, 250, 300],
                 'classifier__max_depth': [None, 10, 20, 30, 50],
                 'classifier__min_samples_split': [2, 5, 10, 15],
                 'classifier__min_samples_leaf': [1, 2, 4, 6],
                 'classifier__max_features': ['sqrt','log2', 0.5, 0.8]
                  }
          rf_grid = model_selection.RandomizedSearchCV(rf_pipe, 
                                                       rf_params,
                                                       n_iter = 250,
                                                       cv = cv, 
                                                       scoring = scorers,
                                                       refit = 'roc_auc',
                                                       random_state = 0)
          warnings.filterwarnings("ignore", message = ".*BaseEstimator._validate_data.*", category = FutureWarning)
          rf_grid.fit(X_train, y_train) 
          cv_results = pd.DataFrame(rf_grid.cv_results_)
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
          rf_cv.append(cv_metrics)
          rf_best = rf_grid.best_estimator_
          rf_metrics = compute_metrics(y_test, rf_best.predict(X_test))        
          rf_test.append(rf_metrics)
    
          print('RF classifier')
          print('--- %s minutes ---' % ((time.time() - start_time) / 60))
          del start_time
    
          # %% ================================================================
          # GB
          print(f'# iteration {i}')
          start_time = time.time()
          gb_clf = GradientBoostingClassifier(random_state = 0)
          gb_pipe = Pipeline([
            ('preprocess', preprocessing),
            ('smote', SMOTE(random_state = 0)),
            ('feature_selection', feature_selection.SelectFromModel(LassoCV(max_iter = 10000,
                                                                            tol = 1e-3,
                                                                            cv = 4, 
                                                                            random_state = 0))), 
            ('classifier', gb_clf)
            ], verbose = False)
          gb_params = {
                 'feature_selection__threshold': ['median','mean', 0.0],
                 'feature_selection__estimator__alphas': [np.logspace(-2, 1, 10)],
                 'classifier__n_estimators': [50, 100, 150, 200, 250, 300],
                 'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                 'classifier__max_depth': [3, 4, 5, 7],
                 'classifier__min_samples_split': [2, 5, 10, 15],
                 'classifier__min_samples_leaf': [1, 2, 4, 6],
                 'classifier__subsample': [0.6, 0.8, 1.0]
                  }
          gb_grid = model_selection.RandomizedSearchCV(gb_pipe, 
                                                       gb_params,
                                                       n_iter = 250,
                                                       cv = cv, 
                                                       scoring = scorers,
                                                       refit = 'roc_auc',
                                                       random_state = 0)
          warnings.filterwarnings("ignore", message = ".*BaseEstimator._validate_data.*", category = FutureWarning)
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
          gb_cv.append(cv_metrics)
          gb_best = gb_grid.best_estimator_
          gb_metrics = compute_metrics(y_test, gb_best.predict(X_test))        
          gb_test.append(gb_metrics)
          print('GB classifier')
          print('--- %s minutes ---' % ((time.time() - start_time) / 60))
          del start_time
    
          # %% ================================================================
          # XGB
          print(f'# iteration {i}')
          start_time = time.time()
          xgb_clf = XGBClassifier(objective = 'binary:logistic', 
                                  booster = 'gbtree', 
                                  eval_metric = 'auc',
                                  tree_method ='hist', 
                                  grow_policy = 'lossguide', 
                                  device = 'cuda',
                                  random_state = 0)
          xgb_pipe = Pipeline([
              ('preprocess', preprocessing),
              ('smote', SMOTE(random_state = 0)),
              # ('feature_selection', feature_selection.SelectFromModel(LassoCV(max_iter = 10000,
              #                                                                 tol = 1e-3,
              #                                                                 cv = 4, 
              #                                                                 random_state = 0))), 
              ('classifier', xgb_clf)
              ], verbose = False)
          xgb_params = {
                  'feature_selection__threshold': ['median','mean', 0.0],
                  'feature_selection__estimator__alphas': [np.logspace(-2, 1, 10)],
                  'classifier__subsample': np.linspace(0.6, 1.0, 5),
                  'classifier__colsample_bytree': np.linspace(0.6, 1.0, 5),
                  'classifier__max_depth': [3, 4, 5, 6, 7],
                  'classifier__learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1],
                  'classifier__min_child_weight': [1, 5, 10, 20],
                  'classifier__gamma': [0, 0.1, 0.3, 0.5],
                  'classifier__reg_alpha': [0, 0.1, 0.5, 1],
                  'classifier__reg_lambda': [0.1, 0.5, 1, 2]
              }
          xgb_grid = model_selection.RandomizedSearchCV(xgb_pipe, 
                                                        xgb_params,
                                                        n_iter = 250,
                                                        cv = cv, 
                                                        scoring = scorers,
                                                        refit = 'roc_auc',
                                                        random_state = 0)
          warnings.filterwarnings("ignore", message = ".*BaseEstimator._validate_data.*", category = FutureWarning)
          xgb_grid.fit(X_train, y_train) 
          cv_results = pd.DataFrame(xgb_grid.cv_results_)
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
          xgb_cv.append(cv_metrics)
          xgb_best = xgb_grid.best_estimator_
          xgb_metrics = compute_metrics(y_test, xgb_best.predict(X_test))        
          xgb_test.append(xgb_metrics)
    
          print('XGB classifier')
          print('--- %s minutes ---' % ((time.time() - start_time) / 60))
          del start_time
    
          # %% ================================================================
          # SVM
          print(f'# iteration {i}')
          start_time = time.time()
          svm_clf = SVC(class_weight = 'balanced', random_state = 0)
          svm_pipe = Pipeline([
              ('preprocess', preprocessing),
              ('smote', SMOTE(random_state = 0)),
              ('feature_selection', feature_selection.SelectFromModel(LassoCV(max_iter = 10000,
                                                                              tol = 1e-3,
                                                                              cv = 4, 
                                                                              random_state = 0))), 
              ('classifier', svm_clf)
              ], verbose = False)
          svm_params = {
                  'feature_selection__threshold': ['median','mean', 0.0],
                  'feature_selection__estimator__alphas': [np.logspace(-2, 1, 10)],
                  'classifier__C': np.arange(0.5, 5, 0.5),
                  'classifier__kernel': ['linear','rbf','poly','sigmoid'],
                  'classifier__gamma' : np.logspace(-4, 1, 10),
                  }
          svm_grid = model_selection.GridSearchCV(svm_pipe, 
                                                  svm_params, 
                                                  cv = cv, 
                                                  scoring = scorers,
                                                  refit = 'roc_auc')
          warnings.filterwarnings("ignore", message = ".*BaseEstimator._validate_data.*", category = FutureWarning)
          svm_grid.fit(X_train, y_train) 
          cv_results = pd.DataFrame(svm_grid.cv_results_)
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
          svm_cv.append(cv_metrics)
          svm_best = svm_grid.best_estimator_
          svm_metrics = compute_metrics(y_test, svm_best.predict(X_test))        
          svm_test.append(svm_metrics)
    
          print('SVM classifier')
          print('--- %s minutes ---' % ((time.time() - start_time) / 60))
          del start_time
    
          # %% ================================================================
          # Logistic Regression
          print(f'# iteration {i}')
          start_time = time.time()
          lr_clf = LogisticRegression(penalty = 'elasticnet', 
                                      solver = 'saga', 
                                      max_iter = 10000, 
                                      class_weight = 'balanced', 
                                      random_state = 0)
          lr_pipe = Pipeline([
             ('preprocess', preprocessing),
             ('smote', SMOTE(random_state = 0)),
             ('feature_selection', feature_selection.SelectFromModel(LassoCV(max_iter = 10000,
                                                                             tol = 1e-3,
                                                                             cv = 4, 
                                                                             random_state = 0))), 
             ('classifier', lr_clf)
             ], verbose = False)
          lr_params = {
                  'feature_selection__threshold': ['mean', 'median', 0.0],
                  'feature_selection__estimator__alphas': [np.logspace(-2, 1, 10)],
                  'classifier__C': np.arange(0.1, 5, 0.1),
                  'classifier__l1_ratio': np.linspace(0.1, 0.9, 9)
                  }
          lr_grid = model_selection.GridSearchCV(lr_pipe, 
                                                 lr_params, 
                                                 cv = cv, 
                                                 scoring = scorers,
                                                 refit = 'roc_auc')
          warnings.filterwarnings("ignore", message = ".*BaseEstimator._validate_data.*", category = FutureWarning)
          lr_grid.fit(X_train, y_train) 
          cv_results = pd.DataFrame(lr_grid.cv_results_)
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
          lr_cv.append(cv_metrics)
          lr_best = lr_grid.best_estimator_
          lr_metrics = compute_metrics(y_test, lr_best.predict(X_test))        
          lr_test.append(lr_metrics)
    
          print('LR classifier')
          print('--- %s minutes ---' % ((time.time() - start_time) / 60))
          del start_time
    
          # %% ================================================================
          # KNN
          print(f'# iteration {i}')
          start_time = time.time()
          knn_clf = KNeighborsClassifier()
          knn_pipe = Pipeline([
              ('preprocess', preprocessing),
              ('smote', SMOTE(random_state = 0)),
              ('feature_selection', feature_selection.SelectFromModel(LassoCV(max_iter = 10000,
                                                                      tol = 1e-3,
                                                                      cv = 4, 
                                                                      random_state = 0))), 
              ('classifier', knn_clf)
          ], verbose = False)
          knn_params = {
                  'feature_selection__threshold': ['median','mean', 0.0],
                  'feature_selection__estimator__alphas': [np.logspace(-2, 1, 10)],
                  'classifier__weights': ['uniform', 'distance'],
                  'classifier__n_neighbors': np.arange(1, 50, 2),
                  'classifier__leaf_size': np.arange(10, 100, 10),
                  'classifier__p': [1, 2]
                  }
          knn_grid = model_selection.GridSearchCV(knn_pipe, 
                                            knn_params,
                                            cv = cv, 
                                            scoring = scorers,
                                            refit = 'roc_auc')
          warnings.filterwarnings("ignore", message = ".*BaseEstimator._validate_data.*", category = FutureWarning)
          knn_grid.fit(X_train, y_train) 
          cv_results = pd.DataFrame(knn_grid.cv_results_)
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
          knn_cv.append(cv_metrics)
          knn_best = knn_grid.best_estimator_
          knn_metrics = compute_metrics(y_test, knn_best.predict(X_test))        
          knn_test.append(knn_metrics)
    
          print('KNN classifier')
          print('--- %s minutes ---' % ((time.time() - start_time) / 60))
          del start_time
    
          # %% ================================================================
          # LGBM
          print(f'# iteration {i}')
          start_time = time.time()
          lgbm_clf = LGBMClassifier(objective = 'binary',
                              boosting_type = 'gbdt',
                              metric = 'auc',
                              device = 'gpu',
                              verbosity = -1,
                              gpu_use_dp = False,
                              max_bin = 255,
                              random_state = 0)
          lgbm_pipe = Pipeline([
              ('preprocess', preprocessing),
              ('smote', SMOTE(random_state = 0)),
              ('feature_selection', feature_selection.SelectFromModel(LassoCV(max_iter = 10000,
                                                                        tol = 1e-3,
                                                                        cv = 4, 
                                                                        random_state = 0))), 
              ('classifier', lgbm_clf)
          ], verbose = False)
          lgbm_params = {
              'feature_selection__threshold': ['mean', 'median', 0.01, 0.02],
              'classifier__subsample': np.linspace(0.6, 1.0, 5),
              'classifier__colsample_bytree': np.linspace(0.6, 1.0, 5),
              'classifier__max_depth': [3, 4, 5, 6, 7],
              'classifier__learning_rate':[0.01, 0.025, 0.05, 0.075, 0.1],
              'classifier__min_child_samples': [5, 10, 20, 50],
              'classifier__reg_alpha': [0, 0.1, 0.5, 1],
              'classifier__reg_lambda': [0, 0.1, 0.5, 1]
              }
          lgbm_grid = model_selection.RandomizedSearchCV(lgbm_pipe,
                                                   lgbm_params,
                                                   n_iter = 250,
                                                   cv = cv,
                                                   scoring = scorers,
                                                   refit = 'roc_auc',
                                                   random_state = 0)
          warnings.filterwarnings("ignore", message = ".*BaseEstimator._validate_data.*", category = FutureWarning)
          lgbm_grid.fit(X_train, y_train) 
          cv_results = pd.DataFrame(lgbm_grid.cv_results_)
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
          lgbm_cv.append(cv_metrics)
          lgbm_best = lgbm_grid.best_estimator_
          lgbm_metrics = compute_metrics(y_test, lgbm_best.predict(X_test))        
          lgbm_test.append(lgbm_metrics)
    
          print('LGBM classifier')
          print('--- %s minutes ---' % ((time.time() - start_time) / 60))
          del start_time
          
          # %% ================================================================
          # StackingClassifier
          estimators = [
                  ('rf', lgbm_best),
                  ('gb', gb_best),
                  ('xgb', xgb_best),
                  ('svm', svm_best),
                  ('lr', lr_best),
                  ('knn', knn_best),
                  ('lgbm', lgbm_best)]
          StClf = StackingClassifier(estimators = estimators,
                                     cv = cv,
                                     stack_method = 'predict')
          cv_scores = model_selection.cross_validate(
              StClf,
              X_train,
              y_train,
              cv = cv,
              scoring = scorers,
              n_jobs = -1,
              return_train_score = False)

          cv_metrics = {k.replace('test_', ''): v.mean()
                        for k, v in cv_scores.items() if k.startswith('test_')}
          StClf_cv.append(cv_metrics)
          StClf.fit(X_train, y_train)
          y_pred = StClf.predict(X_test)
          StClf_metrics = compute_metrics(y_test, y_pred)
          StClf_test.append(StClf_metrics)
          
    return {'i': i,
            'null_test': null_metrics,
            'rf_cv': rf_cv,
            'rf_test': rf_test,
            'gb_cv': gb_cv,
            'gb_test': gb_test,
            'xgb_cv': xgb_cv,
            'xgb_test': xgb_test,
            'svm_cv': svm_cv,
            'svm_test': svm_test,
            'lr_cv': lr_cv,
            'lr_test': lr_test,
            'knn_cv': knn_cv,
            'knn_test': knn_test,
            'lgbm_cv': lgbm_cv,
            'lgbm_test': lgbm_test,
            'StClf_cv': StClf_cv,
            'StClf_test': StClf_test}

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
  #  best_idx = cv_results['mean_test_npv'].idxmax()
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
    
    # p_nsf = 1 - calibrated_clf.predict_proba(X_test)[:,1]
    # thresholds = np.linspace(0.45, 0.85, 100)  
    # y_fail = (y_test.astype(int) == 0).astype(int)
    # nb_model = np.array([net_benefit(y_fail, p_nsf, pt) for pt in thresholds])
    # nsf_rate = y_fail.mean()
    # nb_all  = nsf_rate - (1 - nsf_rate) * (thresholds / (1 - thresholds)) 
    # nb_none = np.zeros_like(thresholds)                                    
        
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
            # 'nb_model': nb_model,
            # 'nb_all': nb_all,
            # 'nb_none': nb_none,
            # 'thresholds': thresholds
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
  #  best_idx = cv_results['mean_test_npv'].idxmax()
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
    
    # p_nsf = 1 - calibrated_clf.predict_proba(X_test)[:,1]
    # thresholds = np.linspace(0.45, 0.85, 100)  
    # y_fail = (y_test.astype(int) == 0).astype(int)
    # nb_model = np.array([net_benefit(y_fail, p_nsf, pt) for pt in thresholds])
    # nsf_rate = y_fail.mean()
    # nb_all  = nsf_rate - (1 - nsf_rate) * (thresholds / (1 - thresholds)) 
    # nb_none = np.zeros_like(thresholds)                                    
        
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
            # 'nb_model': nb_model,
            # 'nb_all': nb_all,
            # 'nb_none': nb_none,
            # 'thresholds': thresholds
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
  #  nb_model_sets = []
  #  nb_all_sets = []
  #  nb_none_sets = []
  #  thresholds_sets = []
    
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
 #   nb_model_sets.extend([res['nb_model'] for res in results])
 #   nb_all_sets.extend([res['nb_all'] for res in results])
 #   nb_none_sets.extend([res['nb_none'] for res in results])
 #   thresholds_sets.extend([res['thresholds'] for res in results])

    results_all = pd.concat([
         expand_metrics(report_null,'null model',modality),
         expand_metrics(report_grid_model, 'model', modality),
         expand_metrics(report_cv_model, 'cv', modality)
    ], ignore_index = True)
    
    features = df.columns[2:]
    fs_feature_importance(test_sets, shap_values_sets, features, metric, modality, site = 0)
   # fs_net_benefit(nb_model_sets, nb_all_sets, nb_none_sets, thresholds_sets)
    
    # import pickle
    # with open(f'shap_values_sets_{modality}.pkl', 'wb') as f:
    #     pickle.dump(shap_values_sets, f)
    # with open(f'test_sets_{modality}.pkl', 'wb') as f:
    #     pickle.dump(test_sets, f)
    # with open(f'odds_ratio_sets_{modality}.pkl', 'wb') as f:
    #     pickle.dump(odds_ratio_sets, f)
    
    return results_all, shap_values_sets, odds_ratio_sets, auc_ci_sets, npv_ci_sets, features

def StClf_classifier_cv(df_FC_st, df_SC_st):
    
    import os
    import warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    
    import numpy as np    
    import pandas as pd
    from sklearn import model_selection
    from sklearn import preprocessing
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    df_FC_BC = df_FC_st[df_FC_st['GT_metrics'] == 'BC'].reset_index(drop = True)
    df_FC_PC = df_FC_st[df_FC_st['GT_metrics'] == 'PC'].reset_index(drop = True)
    df_FC_EC = df_FC_st[df_FC_st['GT_metrics'] == 'EC'].reset_index(drop = True)

    df_SC_BC = df_SC_st[df_SC_st['GT_metrics'] == 'BC'].reset_index(drop = True)
    df_SC_PC = df_SC_st[df_SC_st['GT_metrics'] == 'PC'].reset_index(drop = True)
    df_SC_EC = df_SC_st[df_SC_st['GT_metrics'] == 'EC'].reset_index(drop = True)
     
    X_FC_BC, y = StClf_preprocessing(df_FC_BC)
    X_FC_PC, y = StClf_preprocessing(df_FC_PC)
    X_FC_EC, y = StClf_preprocessing(df_FC_EC)

    X_SC_BC, y = StClf_preprocessing(df_SC_BC)
    X_SC_PC, y = StClf_preprocessing(df_SC_PC)
    X_SC_EC, y = StClf_preprocessing(df_SC_EC)
    
    all_metrics_meta = []
            
    for i in range(0,100):
        
        idx_train, idx_test = model_selection.train_test_split(df_FC_BC.index, test_size = 0.25, stratify = y, random_state = i)
        X_train_FC_BC, X_test_FC_BC = X_FC_BC[idx_train, :], X_FC_BC[idx_test, :]
        X_train_FC_PC, X_test_FC_PC = X_FC_PC[idx_train, :], X_FC_PC[idx_test, :]
        X_train_FC_EC, X_test_FC_EC = X_FC_EC[idx_train, :], X_FC_EC[idx_test, :]
 
        X_train_SC_BC, X_test_SC_BC = X_SC_BC[idx_train, :], X_SC_BC[idx_test, :]
        X_train_SC_PC, X_test_SC_PC = X_SC_PC[idx_train, :], X_SC_PC[idx_test, :]
        X_train_SC_EC, X_test_SC_EC = X_SC_EC[idx_train, :], X_SC_EC[idx_test, :]

        y_train = np.ravel(y[idx_train])
        y_test = np.ravel(y[idx_test])

        print(f'# iteration {i}')
 #       print('FC BC')
 #       pred_train_FC_BC, pred_test_FC_BC = StClf_prediction_cv(X_train_FC_BC, y_train, X_test_FC_BC, y_test, 'FC_BC')
        
        print('FC PC')
        pred_train_FC_PC, pred_test_FC_PC = StClf_prediction_cv(X_train_FC_PC, y_train, X_test_FC_PC, y_test, 'FC_PC')
        
 #       print('FC EC')
 #       pred_train_FC_EC, pred_test_FC_EC = StClf_prediction_cv(X_train_FC_EC, y_train, X_test_FC_EC, y_test, 'FC_EC')
        
 #       print('SC BC')
 #       pred_train_SC_BC, pred_test_SC_BC = StClf_prediction_cv(X_train_SC_BC, y_train, X_test_SC_BC, y_test, 'SC_BC')
        
        print('SC PC')
        pred_train_SC_PC, pred_test_SC_PC = StClf_prediction_cv(X_train_SC_PC, y_train, X_test_SC_PC, y_test, 'SC_PC')
        
 #       print('SC EC')
 #       pred_train_SC_EC, pred_test_SC_EC = StClf_prediction_cv(X_train_SC_EC, y_train, X_test_SC_EC, y_test, 'SC_EC')
        
        pred_train_meta = np.column_stack((pred_train_FC_PC, pred_train_SC_PC))
        pred_test_meta = np.column_stack((pred_test_FC_PC, pred_test_SC_PC))
        pipe_meta = Pipeline([
                    ('scaler', preprocessing.StandardScaler()),
                    ('smote', SMOTE(random_state = 0)),
                    ('meta',  LogisticRegression(penalty = 'elasticnet',
                                                 solver = 'saga',
                                                 max_iter = 10000,
                                                 class_weight = 'balanced',
                                                 random_state = 0))
                    ])
        params_meta = {
                      'meta__C': np.arange(0.1, 5, 0.1),
                      'meta__l1_ratio': np.linspace(0.1, 0.9, 5)
                      }
        grid_meta = model_selection.GridSearchCV(estimator = pipe_meta,
                                                 param_grid = params_meta,
                                                 cv = model_selection.StratifiedKFold(n_splits = 4, 
                                                                                      shuffle = True, 
                                                                                      random_state = 0),
                                                 scoring = 'roc_auc',
                                                 n_jobs = -1,
                                                 refit = True) 
        grid_meta.fit(pred_train_meta, y_train)
        best_meta = grid_meta.best_estimator_
        odds_ratios = np.exp(best_meta.named_steps['meta'].coef_[0])
        
        odds_df = pd.DataFrame({
                                'feature': [
                                            'FC_PC',#'FC_PC','FC_EC',
                                            'SC_PC',#'SC_PC','SC_EC'
                                            ],
                                'odds_ratio': odds_ratios}).sort_values(by = 'odds_ratio', 
                                                                        ascending = False)
        metrics_meta = compute_metrics(y_test, best_meta.predict(pred_test_meta))    
        all_metrics_meta.append({
                                **metrics_meta,
                                'odds_FC_PC': odds_df.loc[odds_df['feature'] =='FC_PC','odds_ratio'].item(),
                            #    'odds_FC_PC': odds_df.loc[odds_df['feature'] =='FC_PC','odds_ratio'].item(),
                            #    'odds_FC_EC': odds_df.loc[odds_df['feature'] =='FC_EC','odds_ratio'].item(),
                                'odds_SC_PC': odds_df.loc[odds_df['feature'] =='SC_PC','odds_ratio'].item(),
                            #    'odds_SC_PC': odds_df.loc[odds_df['feature'] =='SC_PC','odds_ratio'].item(),
                            #    'odds_SC_EC': odds_df.loc[odds_df['feature'] =='SC_EC','odds_ratio'].item(),
                                })
    df_meta_metrics = pd.DataFrame(all_metrics_meta)
    return df_meta_metrics
       
def StClf_preprocessing(df):
    
    import numpy as np
    from sklearn import feature_selection, preprocessing
    
    numerical_features_dc = ['Age onset','Epilepsy duration','Postsurgical duration']
    numerical_index_dc = [list(df.columns).index(col) for col in numerical_features_dc]
    numerical_columns = list(df.columns[13:])
    numerical_index = [list(df.columns).index(col) for col in numerical_columns]
          
    categorical_columns = ['Lesion','Type of surgery']
    categorical_index = [list(df.columns).index(col) for col in categorical_columns]
          
    X_temp = df[numerical_columns].to_numpy()
    corr_matrix = np.corrcoef(X_temp, rowvar = False)
    to_remove = set()

    for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            if abs(corr_matrix[i, j]) > 0.85:
                to_remove.add(numerical_columns[j])
    del i,j
          
    df_removed = df.drop(columns = to_remove)
    numerical_columns_removed = list(df_removed.iloc[:, 13:].columns)
    numerical_index_removed = [df_removed.columns.get_loc(col) for col in numerical_columns_removed]
          
    X, y = df_removed.to_numpy(), df_removed['Status'].to_numpy()
    selector_num = feature_selection.SelectFpr(score_func = feature_selection.f_classif, alpha = 0.05)
          
    X_num = selector_num.fit_transform(X[:, numerical_index_removed], y)
    X_num = np.concatenate((X[:, numerical_index_dc], X_num), axis = 1)
    features = categorical_columns + numerical_features_dc + \
        [col for col, keep in zip(numerical_columns_removed, selector_num.get_support()) if keep]

    encoder = preprocessing.OneHotEncoder(sparse_output = False, handle_unknown = 'ignore', drop = 'first')
    X_cat = encoder.fit_transform(X[:, categorical_index])
    X_selected = np.concatenate((X_cat, X_num), axis = 1)
    
    return X_selected, y
          
def StClf_prediction_cv(X_train, y_train, X_test, y_test, modality):
      
    import numpy as np
    from sklearn import preprocessing, feature_selection, model_selection
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, LassoCV
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    import time
    
    start_time = time.time()
    if modality == 'FC_PC' or modality == 'FC_EC':
        knn_clf = KNeighborsClassifier()
        knn_pipe = Pipeline([
           ('scaler', preprocessing.StandardScaler()),
           ('smote', SMOTE(random_state = 0)),
           ('feature_selection', feature_selection.SelectFromModel(LassoCV(max_iter = 10000,
                                                                           tol = 1e-3,
                                                                           cv = 4, 
                                                                           random_state = 0))), 
           ('classifier', knn_clf)
        ], verbose = False)
        knn_params = {
                      'feature_selection__threshold': ['median','mean', 0.0],
                      'feature_selection__estimator__alphas': [np.logspace(-2, 1, 10)],
                      'classifier__weights': ['uniform', 'distance'],
                      'classifier__n_neighbors': np.arange(1, 50, 2),
                      'classifier__leaf_size': np.arange(10, 100, 10),
                      'classifier__p': [1, 2]
                      }
        knn_grid = model_selection.GridSearchCV(knn_pipe, 
                                                knn_params,
                                                cv = model_selection.StratifiedKFold(n_splits = 4, 
                                                                                     shuffle = True, 
                                                                                     random_state = 0), 
                                                scoring = 'roc_auc',
                                                n_jobs = -1)
        pred_train = model_selection.cross_val_predict(estimator = knn_grid,
                                                       X = X_train,
                                                       y = y_train,
                                                       cv = model_selection.StratifiedKFold(n_splits = 4, 
                                                                                            shuffle = True, 
                                                                                            random_state = 0), 
                                                       method = 'predict_proba',
                                                       n_jobs = -1)[:,1]
        knn_grid.fit(X_train, y_train)
        pred_test = knn_grid.predict_proba(X_test)[:, 1]
        print('KNN classifier')
        
    elif modality == 'SC_PC':
        rf_clf = RandomForestClassifier(random_state = 0)
        rf_pipe = Pipeline([
            ('scaler', preprocessing.StandardScaler()),
            ('smote', SMOTE(random_state = 0)),
            ('feature_selection', feature_selection.SelectFromModel(LassoCV(max_iter = 10000,
                                                                            tol = 1e-3,
                                                                            cv = 4, 
                                                                            random_state = 0))), 
            ('classifier', rf_clf)
        ], verbose = False)
        rf_params = {
                     'feature_selection__threshold': ['median', 'mean', 0.0],
                     'feature_selection__estimator__alphas': [np.logspace(-2, 1, 10)],
                     'classifier__n_estimators': [50, 100, 150, 200, 250, 300],
                     'classifier__max_depth': [None, 10, 20, 30, 50],
                     'classifier__min_samples_split': [2, 5, 10, 15],
                     'classifier__min_samples_leaf': [1, 2, 4, 6],
                     'classifier__max_features': ['sqrt', 'log2', 0.5, 0.8]
                      }
        rf_grid = model_selection.RandomizedSearchCV(rf_pipe, 
                                                     rf_params,
                                                     n_iter = 250,
                                                     cv = model_selection.StratifiedKFold(n_splits = 4, 
                                                                                          shuffle = True, 
                                                                                          random_state = 0), 
                                                     scoring = 'roc_auc',
                                                     random_state = 0)
        pred_train = model_selection.cross_val_predict(estimator = rf_grid,
                                                       X = X_train,
                                                       y = y_train,
                                                       cv = model_selection.StratifiedKFold(n_splits = 4, 
                                                                                            shuffle = True, 
                                                                                            random_state = 0), 
                                                       method = 'predict_proba',
                                                       n_jobs = -1)[:,1]
        rf_grid.fit(X_train, y_train)
        pred_test = rf_grid.predict_proba(X_test)[:, 1]
        print('RF classifier')
    else:            
        lr_clf = LogisticRegression(penalty = 'elasticnet', 
                                    solver = 'saga', 
                                    max_iter = 10000, 
                                    class_weight = 'balanced',
                                    random_state = 0)
        lr_pipe = Pipeline([
            ('scaler', preprocessing.StandardScaler()),
            ('smote', SMOTE(random_state = 0)),
            ('feature_selection', feature_selection.SelectFromModel(LassoCV(max_iter = 10000,
                                                                            tol = 1e-3,
                                                                            cv = 4, 
                                                                            random_state = 0))), 
            ('classifier', lr_clf)
            ], verbose = False)
        lr_params = {
            'feature_selection__threshold': ['mean', 'median', 0.0],
            'feature_selection__estimator__alphas': [np.logspace(-2, 1, 10)],
            'classifier__C': np.arange(0.1, 5, 0.1),
            'classifier__l1_ratio': np.linspace(0.1, 0.9, 9)
            }
        lr_grid = model_selection.GridSearchCV(lr_pipe, 
                                               lr_params, 
                                               cv = model_selection.StratifiedKFold(n_splits = 4, 
                                                                                    shuffle = True, 
                                                                                    random_state = 0), 
                                               scoring = 'roc_auc',
                                               n_jobs = -1)
        pred_train = model_selection.cross_val_predict(estimator = lr_grid,
                                                       X = X_train,
                                                       y = y_train,
                                                       cv = model_selection.StratifiedKFold(n_splits = 4, 
                                                                                            shuffle = True, 
                                                                                            random_state = 0), 
                                                       method = 'predict_proba',
                                                       n_jobs = -1)[:,1]
        lr_grid.fit(X_train, y_train)
        pred_test = lr_grid.predict_proba(X_test)[:, 1]
        print('LR classifier')
    
    print('--- %s minutes ---' % ((time.time() - start_time) / 60))
    del start_time
    return pred_train, pred_test
        