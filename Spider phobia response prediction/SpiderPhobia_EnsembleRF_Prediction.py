# -*- coding: utf-8 -*-
"""
Created between May and August 2022
by authors Alice Chavanne, Charlotte Meinke and Kevin Hilbert
"""
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import sys
import pickle
import csv
import multiprocessing
import os
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
from sklearn import set_config
from sklearn.dummy import DummyClassifier
from multiprocessing import Pool
from sklearn.metrics import roc_curve, auc
import time
import shap


shap.initjs()
start_time = time.time()


set_config(working_memory = 100000)
print(multiprocessing.cpu_count())

# Data names and path
STANDARDPATH = "Path\\To\\Directory\\Containing\\Feature\\Files\\"
OPTIONS = {'name_model': 'RF'}
OPTIONS['name_features'] =  ['demographics.txt','activation_30ROIs.txt','gppi_phasic_30ROIs.txt','gppi_sustained_30ROIs.txt','graph_gppi_phasic_30ROIs.txt','graph_gppi_sustained_30ROIs.txt', 'structural.txt', 'variability_30ROIs.txt']
OPTIONS['names_1stlevel_classifiers'] = ['demo', 'activation', 'gPPI_phasic', 'gPPI_sustained', 'graph_gPPI_phasic','graph_gPPI_sustained', 'structural', 'variability']
OPTIONS['abbreviations_features'] = ['_feat_demo','_feat_activation','_feat_gppi_phasic','_feat_gppi_sustained','_feat_graph_gppi_phasic','_feat_graph_gppi_sustained','_feat_structural','_feat_variability','_softmax_voting','meta_learner_2nd_lvl_RF']
OPTIONS['name_labels'] = 'labels.txt'

# Overall options

OPTIONS['number_iterations'] = 100
OPTIONS['test_size_option'] = 0.2
OPTIONS['threshold_option'] = "mean" #For lasso feature selection, threshold for feature importance. Choose either "mean", or "median", or float value, or mix (for ex: "1.25*mean")
OPTIONS['dummy_option'] = 1 # 0 = no dummy outputs, 1 = dummy outputs with blind classifier always predicting majority class are created to compare
OPTIONS['shap'] = 1 # 1 compute shap values, 0 no


def prepare_data(numrun):
    
    global STANDARDPATH, OPTIONS
    
    random_state_seed = numrun

    
    for model in range(len(OPTIONS['name_features'])):

        # Import raw data and labels
        
        features_import_path = STANDARDPATH + OPTIONS['name_features'][model]
        labels_import_path = STANDARDPATH + OPTIONS['name_labels']
        features_import = read_csv(features_import_path, sep=",", header=0)
        labels_import = read_csv(labels_import_path, sep=",", header=0)
         
        features_import = features_import.drop(columns="Subjects")
        labels_import = labels_import.drop(columns="Subjects")
        all_feature_names = features_import.columns


        # Split train / test sets

        X_train, X_test, y_train, y_test = train_test_split(features_import, labels_import, stratify=None, test_size=OPTIONS['test_size_option'], random_state=random_state_seed)

        y_train= np.array(y_train)
        y_test= np.array(y_test)
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        save_cv_option_features_train = STANDARDPATH + 'features_train' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][model] + '_save_cv_fold_'
        save_cv_option_features_test =  STANDARDPATH + 'features_test' + os.sep + OPTIONS['name_model']+ OPTIONS['abbreviations_features'][model] + '_save_cv_fold_'
        save_cv_option_labels_train =  STANDARDPATH + 'labels_train' + os.sep + OPTIONS['name_model']+ OPTIONS['abbreviations_features'][model] + '_save_cv_fold_'
        save_cv_option_labels_test =  STANDARDPATH + 'labels_test' + os.sep + OPTIONS['name_model']+ OPTIONS['abbreviations_features'][model] + '_save_cv_fold_'

        os.makedirs(os.path.dirname(STANDARDPATH + 'features_train' + os.sep), exist_ok=True)
        os.makedirs(os.path.dirname(STANDARDPATH + 'features_test' + os.sep), exist_ok=True)
        os.makedirs(os.path.dirname(STANDARDPATH + 'labels_train' + os.sep), exist_ok=True)
        os.makedirs(os.path.dirname(STANDARDPATH + 'labels_test' + os.sep), exist_ok=True)

        full_path_cv_option = save_cv_option_features_train + str (random_state_seed) + '_features_train.txt'
        with open(full_path_cv_option, 'w+', newline='') as file:
            csvsave = csv.writer(file, delimiter=' ')
            csvsave.writerows(X_train)

        full_path_cv_option = save_cv_option_labels_train + str (random_state_seed) + '_labels_train.txt'
        with open(full_path_cv_option, 'w+', newline='') as file:
            csvsave = csv.writer(file, delimiter=' ')
            csvsave.writerows(y_train)
            
        full_path_cv_option = save_cv_option_features_test + str (random_state_seed) + '_features_test.txt'
        with open(full_path_cv_option, 'w+', newline='') as file:
            csvsave = csv.writer(file, delimiter=' ')
            csvsave.writerows(X_test)
            
        full_path_cv_option = save_cv_option_labels_test + str (random_state_seed) + '_labels_test.txt'
        with open(full_path_cv_option, 'w+', newline='') as file:
            csvsave = csv.writer(file, delimiter=' ')
            csvsave.writerows(y_test)

        full_featurenames_option = STANDARDPATH + 'features_train' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][model] +'all_feature_names.txt'
        with open(full_featurenames_option, 'w+', newline='') as file:
            for line in all_feature_names:
                file.write(line + '\n')


def accuracies_metrics_for_classifiers(predictions, labels_test_set):

    y_prediction = predictions

    counter_class1_correct = 0
    counter_class2_correct = 0
    counter_class1_incorrect = 0
    counter_class2_incorrect = 0

    for i in range(len(labels_test_set)):
        if y_prediction[i,0] == y_prediction[i,1]:
            y_prediction[i,2] = 1
            if y_prediction[i,1] == 1:
                counter_class1_correct = counter_class1_correct + 1
            else: 
                counter_class2_correct = counter_class2_correct + 1
        else:
            y_prediction[i,2] = 0
            if y_prediction[i,1] == 1:
                counter_class1_incorrect = counter_class1_incorrect + 1
            else: 
                counter_class2_incorrect = counter_class2_incorrect + 1

    accuracy = y_prediction.mean(axis=0)[2]
    accuracy_class1 = counter_class1_correct / (counter_class1_correct + counter_class1_incorrect)
    accuracy_class2 = counter_class2_correct / (counter_class2_correct + counter_class2_incorrect)
    balanced_accuracy = (accuracy_class1 + accuracy_class2) / 2

    return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy


def classification_pipeline(numrun):

    global STANDARDPATH, OPTIONS
    
    print(numrun)
    random_state_seed = numrun
    os.makedirs(os.path.dirname(STANDARDPATH + 'metalearner_input' + os.sep), exist_ok=True)

    load_cv_option_cur_model =  STANDARDPATH + 'metalearner_input' + os.sep + 'current_model.txt'

    with open(load_cv_option_cur_model, "rb") as input_file:
        current_model = int(pickle.load(input_file))
     

    # Import split data and labels

    full_path_cv_option_features_train = STANDARDPATH + 'features_train' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_save_cv_fold_'+ str (random_state_seed) + '_features_train.txt'
    full_path_cv_option_labels_train = STANDARDPATH + 'labels_train' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_save_cv_fold_' + str (random_state_seed) + '_labels_train.txt'
    full_path_cv_option_features_test = STANDARDPATH + 'features_test' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_save_cv_fold_' + str (random_state_seed) + '_features_test.txt'
    full_path_cv_option_labels_test = STANDARDPATH + 'labels_test' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_save_cv_fold_' + str (random_state_seed) + '_labels_test.txt'
    
    X_train = read_csv(full_path_cv_option_features_train, sep="\s", header=None, engine='python')
    X_test = read_csv(full_path_cv_option_features_test, sep="\s", header=None, engine='python')
    y_train = read_csv(full_path_cv_option_labels_train, sep="\s", header=None, engine='python')
    y_test = read_csv(full_path_cv_option_labels_test, sep="\s", header=None, engine='python')


    # Imputation missing values and scaling of training and test sets

    imp_arith = SimpleImputer(missing_values=999, strategy='mean')
    imp_median = SimpleImputer(missing_values=888, strategy='median')
    imp_mode = SimpleImputer(missing_values=777, strategy='most_frequent')
    imp_arith.fit(X_train)
    imp_median.fit(X_train)
    imp_mode.fit(X_train)
    X_train_imputed = imp_arith.transform(X_train)
    X_train_imputed = imp_median.transform(X_train_imputed)
    X_train_imputed = imp_mode.transform(X_train_imputed)
    X_test_imputed = imp_arith.transform(X_test)
    X_test_imputed = imp_median.transform(X_test_imputed)
    X_test_imputed = imp_mode.transform(X_test_imputed)

    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train_imputed)
    X_train_imputed_scaled = scaler.transform(X_train_imputed)
    X_test_imputed_scaled = scaler.transform(X_test_imputed)

    # Feature selection in training set

    y_train=np.ravel(y_train)
    

    clf_elastic_logregression_features = SGDClassifier(loss='log_loss', penalty='elasticnet', fit_intercept=False, tol=0.0001, max_iter=1000, random_state=random_state_seed)
    clf_elastic_logregression_features_search = GridSearchCV(estimator = clf_elastic_logregression_features, param_grid ={'l1_ratio': np.arange(0,1,0.1)}, scoring='balanced_accuracy', n_jobs=1, verbose=0)
    clf_elastic_logregression_features_search.fit(X_train_imputed_scaled, y_train)
    sfm = SelectFromModel(clf_elastic_logregression_features_search.best_estimator_, threshold=OPTIONS['threshold_option'], prefit = True)
    X_train_imputed_scaled_selected = sfm.transform(X_train_imputed_scaled)


    #Fit Random Forest classifier


    clf = RandomForestClassifier(n_estimators=1000, criterion = 'gini', max_depth= None, min_samples_split= 2, min_samples_leaf= 1, bootstrap= True, oob_score=True, class_weight='balanced', random_state=random_state_seed)

    clf = clf.fit(X_train_imputed_scaled_selected, y_train)


    #Feature Selection in test set

    y_test=np.ravel(y_test)

    X_test_scaled_imputed_selected = sfm.transform(X_test_imputed_scaled)


    #Prediction in test set
    
    y_prediction = np.zeros((len(y_test), 3))
    
    y_prediction[:,0] = clf.predict(X_test_scaled_imputed_selected)
        
    y_prediction[:,1] = y_test[:]

    meta_learner_input = np.zeros((len(y_test), 4))
    meta_learner_input[:,0] = y_test[:] 
    meta_learner_input[:,1] = clf.predict(X_test_scaled_imputed_selected)
    meta_learner_input[:,2] = clf.predict_proba(X_test_scaled_imputed_selected)[:,0]
    meta_learner_input[:,3] = clf.predict_proba(X_test_scaled_imputed_selected)[:,1]
    
    meta_learner_input_train = np.zeros((len(y_train), 4))
    meta_learner_input_train[:,0] = y_train[:] 
    meta_learner_input_train[:,1] = clf.predict(X_train_imputed_scaled_selected)
    meta_learner_input_train[:,2] = clf.predict_proba(X_train_imputed_scaled_selected)[:,0]
    meta_learner_input_train[:,3] = clf.predict_proba(X_train_imputed_scaled_selected)[:,1]

    accuracy, accuracy_class1, accuracy_class2, balanced_accuracy = accuracies_metrics_for_classifiers(y_prediction, y_test)
    oob_accuracy = clf.oob_score_
    log_loss_value = log_loss(y_test, clf.predict_proba(X_test_scaled_imputed_selected), normalize=True)


    feature_importances = np.zeros((len(sfm.get_support())))
    counter_features_selected = 0
    for number_features in range(len(sfm.get_support())):
        if sfm.get_support()[number_features] == True:
            feature_importances[number_features] = clf.feature_importances_[counter_features_selected]
            counter_features_selected = counter_features_selected + 1
        else:
            feature_importances[number_features] = 0
    selected_features_bool = sfm.get_support()

    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test_scaled_imputed_selected)[:,1])
    mean_fpr = np.linspace(0, 1, 100)
    tprs = np.interp(mean_fpr, fpr, tpr)
    roc_auc = auc(fpr, tpr)
    fraction_positives, mean_predicted_value = calibration_curve(y_test, clf.predict_proba(X_test_scaled_imputed_selected)[:,1], n_bins=10)


    print(OPTIONS['abbreviations_features'][current_model])
    print(current_model)

    save_cv_option_meta_learner_input =  STANDARDPATH + 'metalearner_input' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_save_cv_fold_'
        
    full_path_cv_option = save_cv_option_meta_learner_input + str (random_state_seed) + '_predictions.txt'
    with open(full_path_cv_option, 'w', newline='') as file:
        csvsave = csv.writer(file, delimiter=' ')
        csvsave.writerows(meta_learner_input)
        
    save_cv_option_meta_learner_input_train =  STANDARDPATH + 'metalearner_input' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_save_cv_fold_'
        
    full_path_cv_option = save_cv_option_meta_learner_input_train + str (random_state_seed) + '_train_predictions.txt'
    with open(full_path_cv_option, 'w', newline='') as file:
        csvsave = csv.writer(file, delimiter=' ')
        csvsave.writerows(meta_learner_input_train)

    save_cv_option_oob_input =  STANDARDPATH + 'metalearner_input' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_save_cv_fold_'
        
    full_path_cv_option = save_cv_option_oob_input + str (random_state_seed) + '_oob_acc.txt'
    with open(full_path_cv_option, 'wb') as AutoPickleFile:
        pickle.dump((oob_accuracy), AutoPickleFile)



    # Calculate SHAP values for test set

    save_option_prebuild = STANDARDPATH + 'metalearner_input' + os.sep
    os.makedirs(os.path.dirname(save_option_prebuild), exist_ok=True)

    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] +'_'+ str (random_state_seed) +'_round_selected_features.txt'
    pd.DataFrame(selected_features_bool).to_csv(save_option, index = None, header = None)
    if OPTIONS['shap'] == 1 :
        explainer = shap.TreeExplainer(model = clf, data = X_train_imputed_scaled_selected, model_output='raw', feature_perturbation='interventional')
        shap_values = explainer.shap_values(X_test_scaled_imputed_selected, check_additivity = False) # check additivity is disabled because is an unresolved shap problem for tree models, for now
    elif OPTIONS['shap'] == 0:
        shap_values = []
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_'+ str (random_state_seed)+'_round_shap_values''n_estimators'
    with open(save_option, 'wb') as AutoPickleFile:
        pickle.dump(shap_values, AutoPickleFile)


    #Dummy classifier

    if OPTIONS['dummy_option'] == 0:
        dummy_acc = 0
    elif OPTIONS['dummy_option'] ==1:
        clf_dummy = DummyClassifier(strategy = 'most_frequent')
        clf_dummy.fit(X_train_imputed_scaled_selected, y_train)
        clf_dummy.predict(X_test_scaled_imputed_selected)
        dummy_acc = clf_dummy.score(X_test_scaled_imputed_selected, y_test)

    return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value, dummy_acc



def save_performance_measures(accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value, dummy_acc):
 
    global STANDARDPATH, OPTIONS
    
    load_cv_option_cur_model =  STANDARDPATH + 'metalearner_input' + os.sep + 'current_model.txt'
    with open(load_cv_option_cur_model, "rb") as input_file:
        current_model = int(pickle.load(input_file))

    save_option_prebuild = STANDARDPATH + 'accuracy' + os.sep
    os.makedirs(os.path.dirname(save_option_prebuild), exist_ok=True)
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_accuracy.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(accuracy)

    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_accuracy_class1.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(accuracy_class1)
            
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_accuracy_class2.txt' 
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(accuracy_class2)
            
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_balanced_accuracy.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(balanced_accuracy)
            
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_oob_accuracy.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(oob_accuracy)
            
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_log_loss.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(log_loss_value)           
               
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_feature_importances.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerows(feature_importances)
            
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_fpr.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(fpr)
            
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_tpr.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(tpr)
            
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_tprs.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(tprs)
            
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_roc_auc.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(roc_auc)
            
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_fraction_positives.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(fraction_positives)
            
    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_predicted_value.txt'   
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(mean_predicted_value)

    save_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_per_round_dummy_acc.txt'
    with open(save_option, 'w', newline='\n') as file:
            csvsave = csv.writer(file, delimiter=',')
            csvsave.writerow(dummy_acc)



def list_to_flatlist(input_data):
    
    accuracy_flat = []
    accuracy_class1_flat = []
    accuracy_class2_flat = []
    balanced_accuracy_flat = []
    oob_accuracy_flat = []
    log_loss_value_flat = []
    feature_importances_flat = []
    fpr_flat = []
    tpr_flat = []
    tprs_flat = []
    roc_auc_flat = []
    fraction_positives_flat = []
    mean_predicted_value_flat = []
    dummy_acc_flat = []

    counter = 0    
    
    for sublist in input_data:
        for itemnumber in range(len(sublist)):
            if itemnumber == 0:
                accuracy_flat.append(sublist[itemnumber])
            elif itemnumber == 1:
                accuracy_class1_flat.append(sublist[itemnumber])
            elif itemnumber == 2:
                accuracy_class2_flat.append(sublist[itemnumber])
            elif itemnumber == 3:
                balanced_accuracy_flat.append(sublist[itemnumber])
            elif itemnumber == 4:
                oob_accuracy_flat.append(sublist[itemnumber])
            elif itemnumber == 5:
                log_loss_value_flat.append(sublist[itemnumber])
            elif itemnumber == 6:
                feature_importances_flat.append([0,0,0])
            elif itemnumber == 7:
                fpr_flat.append(sublist[itemnumber])
            elif itemnumber == 8:
                tpr_flat.append(sublist[itemnumber])
            elif itemnumber == 9:
                tprs_flat.append(sublist[itemnumber])
            elif itemnumber == 10:
                roc_auc_flat.append(sublist[itemnumber])
            elif itemnumber == 11:
                fraction_positives_flat.append(sublist[itemnumber])
            elif itemnumber == 12:
                mean_predicted_value_flat.append(sublist[itemnumber])
            elif itemnumber == 13:
                dummy_acc_flat.append(sublist[itemnumber])

        counter = counter + 1                
        
    return accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat, dummy_acc_flat



def aggregate_scores(accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value):
    
    global STANDARDPATH, OPTIONS
    
    load_cv_option_cur_model =  STANDARDPATH + 'metalearner_input' + os.sep + 'current_model.txt'
    with open(load_cv_option_cur_model, "rb") as input_file:
        current_model = int(pickle.load(input_file))

    accuracy_min = min(accuracy)
    accuracy_max = max(accuracy)
    accuracy_mean = np.mean(accuracy)
    accuracy_std = np.std(accuracy)
    accuracy_class1_min = min(accuracy_class1)
    accuracy_class1_max = max(accuracy_class1)
    accuracy_class1_mean = np.mean(accuracy_class1)
    accuracy_class1_std = np.std(accuracy_class1)
    accuracy_class2_min = min(accuracy_class2)
    accuracy_class2_max = max(accuracy_class2)
    accuracy_class2_mean = np.mean(accuracy_class2)
    accuracy_class2_std = np.std(accuracy_class2)
    balanced_accuracy_min = min(balanced_accuracy)
    balanced_accuracy_max = max(balanced_accuracy)
    balanced_accuracy_mean = np.mean(balanced_accuracy)
    balanced_accuracy_std = np.std(balanced_accuracy)
    oob_accuracy_min = min(oob_accuracy)
    oob_accuracy_max = max(oob_accuracy)
    oob_accuracy_mean = np.mean(oob_accuracy)
    oob_accuracy_std = np.std(oob_accuracy)
    log_loss_value_min = min(log_loss_value)
    log_loss_value_max = max(log_loss_value)
    log_loss_value_mean = np.mean(log_loss_value)
    log_loss_value_std = np.std(log_loss_value)
    
    number_rounds = len(accuracy)
    

    save_option_prebuild = STANDARDPATH + 'accuracy' + os.sep  
    
    savepath_option = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '.txt'   
    f = open(savepath_option, 'w')
    f.write('The scikit-learn version is {}.'.format(sklearn.__version__) + 
             '\nNumber of Rounds: ' + str(number_rounds) + 
             '\nMin Accuracy: ' + str(accuracy_min) + '\nMax Accuracy: ' + str(accuracy_max) + '\nMean Accuracy: ' + str(accuracy_mean) + '\nStd Accuracy: ' + str(accuracy_std) +
             '\nMin Accuracy_class_1: ' + str(accuracy_class1_min) + '\nMax Accuracy_class_1: ' + str(accuracy_class1_max) + '\nMean Accuracy_class_1: ' + str(accuracy_class1_mean) + '\nStd Accuracy_class_1: ' + str(accuracy_class1_std) +
             '\nMin Accuracy_class_2: ' + str(accuracy_class2_min) + '\nMax Accuracy_class_2: ' + str(accuracy_class2_max) + '\nMean Accuracy_class_2: ' + str(accuracy_class2_mean) + '\nStd Accuracy_class_2: ' + str(accuracy_class2_std) +
             '\nMin Balanced_Accuracy: ' + str(balanced_accuracy_min) + '\nMax Balanced_Accuracy: ' + str(balanced_accuracy_max) + '\nMean Balanced_Accuracy: ' + str(balanced_accuracy_mean) + '\nStd Balanced_Accuracy: ' + str(balanced_accuracy_std) +
             '\nMin OOB_Accuracy: ' + str(oob_accuracy_min) + '\nMax OOB_Accuracy: ' + str(oob_accuracy_max) + '\nMean OOB_Accuracy: ' + str(oob_accuracy_mean) + '\nStd OOB_Accuracy: ' + str(oob_accuracy_std) +
             '\nMin Log-Loss: ' + str(log_loss_value_min) + '\nMax Log-Loss: ' + str(log_loss_value_max) + '\nMean Log-Loss: ' + str(log_loss_value_mean) + '\nStd Log-Loss: ' + str(log_loss_value_std)
             )
    f.close()

    print('Number of Rounds: ' + str(number_rounds) + 
             '\nMin Accuracy: ' + str(accuracy_min) + '\nMax Accuracy: ' + str(accuracy_max) + '\nMean Accuracy: ' + str(accuracy_mean) + '\nStd Accuracy: ' + str(accuracy_std) +
             '\nMin Accuracy_class_1: ' + str(accuracy_class1_min) + '\nMax Accuracy_class_1: ' + str(accuracy_class1_max) + '\nMean Accuracy_class_1: ' + str(accuracy_class1_mean) + '\nStd Accuracy_class_1: ' + str(accuracy_class1_std) +
             '\nMin Accuracy_class_2: ' + str(accuracy_class2_min) + '\nMax Accuracy_class_2: ' + str(accuracy_class2_max) + '\nMean Accuracy_class_2: ' + str(accuracy_class2_mean) + '\nStd Accuracy_class_2: ' + str(accuracy_class2_std) +
             '\nMin Balanced_Accuracy: ' + str(balanced_accuracy_min) + '\nMax Balanced_Accuracy: ' + str(balanced_accuracy_max) + '\nMean Balanced_Accuracy: ' + str(balanced_accuracy_mean) + '\nStd Balanced_Accuracy: ' + str(balanced_accuracy_std) +
             '\nMin OOB_Accuracy: ' + str(oob_accuracy_min) + '\nMax OOB_Accuracy: ' + str(oob_accuracy_max) + '\nMean OOB_Accuracy: ' + str(oob_accuracy_mean) + '\nStd OOB_Accuracy: ' + str(oob_accuracy_std) +
             '\nMin Log-Loss: ' + str(log_loss_value_min) + '\nMax Log-Loss: ' + str(log_loss_value_max) + '\nMean Log-Loss: ' + str(log_loss_value_mean) + '\nStd Log-Loss: ' + str(log_loss_value_std)
             )
 

    plt.close('all')
    
    min_mean_predicted_value = np.zeros((len(mean_predicted_value)))
    max_mean_predicted_value = np.zeros((len(mean_predicted_value)))
    
    for j in range(len(fraction_positives)):
        min_mean_predicted_value[j] = min(mean_predicted_value[j])
        max_mean_predicted_value[j] = max(mean_predicted_value[j])
        
    minmin_mean_predicted_value = min(min_mean_predicted_value)
    maxmax_mean_predicted_value = max(max_mean_predicted_value)    
    mean_mean_predicted_value = np.linspace(minmin_mean_predicted_value, maxmax_mean_predicted_value, int((round((maxmax_mean_predicted_value - minmin_mean_predicted_value)*100))))
    fraction_positives_interpolated = np.zeros((len(fraction_positives),int((round((maxmax_mean_predicted_value - minmin_mean_predicted_value)*100)))))
    
    for i in range(len(fraction_positives)):
        if i == 0:
            plt.plot(mean_predicted_value[i], fraction_positives[i], lw=1, color = 'grey', marker='.', label = 'Individual Iterations', alpha=0.3)
        else:
            plt.plot(mean_predicted_value[i], fraction_positives[i], lw=1, color = 'grey', marker='.', alpha=0.3)
            
        fraction_positives_interpolated[i,:] = np.interp(mean_mean_predicted_value, mean_predicted_value[i], fraction_positives[i])

            
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Perfectly calibrated', alpha=.7)
    mean_fraction_positives_interpolated = np.mean(fraction_positives_interpolated, axis=0)
    plt.plot(mean_mean_predicted_value, mean_fraction_positives_interpolated, color='k', label=r'Mean calibration', lw=2, alpha=1)    
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.title('Calibration plots  (reliability curve)')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Count')
    plt.ylabel("Fraction of positives")
    plt.legend(loc="lower right", framealpha = 0.92)
    
    calibrations_path = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '_calibrations.png'
    plt.savefig(calibrations_path, dpi = 300)

    
    plt.close('all')
    
    for i in range(len(fpr)):
        if i == 0:
            plt.plot(fpr[i], tpr[i], lw=1, color = 'grey', label = 'Individual Iterations', alpha=0.3)
        else:
            plt.plot(fpr[i], tpr[i], lw=1, color = 'grey', alpha=0.3)
            
    mean_fpr = np.linspace(0, 1, 100)
    tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    print(mean_tpr)
    mean_tpr[0] = 0
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_auc)
    plt.plot(mean_fpr, mean_tpr, color='k', label=r'Mean ROC', lw=2, alpha=1) #plus minus: $\pm$
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='w', label='AUC = %0.2f, SD = %0.2f' % (mean_auc, std_auc), alpha=.001)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance', alpha=.7)

    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", framealpha = 0.92)
    
    
    roc_path = save_option_prebuild + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][current_model] + '.png'
    plt.savefig(roc_path, dpi = 300)
    plt.show()



def meta_learner_softmax_voting(numrun):

    global STANDARDPATH, OPTIONS
    
    print(numrun)
    random_state_seed = numrun
               
        
    """
    "Import Inputs
    """
    meta_learner_inputs = []
    checked_classifiers = []
    for i_classifier in range(len(OPTIONS['names_1stlevel_classifiers'])):
        full_path_cv_option_features = STANDARDPATH + 'metalearner_input' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][i_classifier] + '_save_cv_fold_'+ str (random_state_seed) + '_predictions.txt'
        checked_classifiers.append(OPTIONS['names_1stlevel_classifiers'][i_classifier])
        meta_learner_inputs.append(np.array(read_csv(full_path_cv_option_features,sep="\s", header=None, engine='python')))

    vote_softmax_raw = np.zeros((len(meta_learner_inputs[0]), 1))
    vote_softmax_raw_2 = np.zeros((len(meta_learner_inputs[0]), 1))
    vote_softmax = np.zeros((len(meta_learner_inputs[0]), 1))
    vote_softmax_result = np.zeros((len(meta_learner_inputs[0]), 1))


    #Softmax Voting

    for z in range(len(meta_learner_inputs[0])):
        vote_softmax_raw[z] = sum(meta_learner_inputs[i_classifier][z,2] for i_classifier in range(len(checked_classifiers)))
        vote_softmax_raw_2[z] = sum(meta_learner_inputs[i_classifier][z,3] for i_classifier in range(len(checked_classifiers))) / len(checked_classifiers)
        if vote_softmax_raw[z] < (len(checked_classifiers)*0.5):
            vote_softmax[z] = 1
        else:
            vote_softmax[z] = 0
        if vote_softmax[z] == meta_learner_inputs[0][z,0]:
            vote_softmax_result[z] = 1
        else:
            vote_softmax_result[z] = 0  


    accuracy_per_subject = vote_softmax_result.mean(axis=1)

    counter_class1_correct = 0
    counter_class2_correct = 0
    counter_class1_incorrect = 0
    counter_class2_incorrect = 0
    
    for i in range(len(accuracy_per_subject)):
        if accuracy_per_subject[i] == 1:
            if meta_learner_inputs[0][i,0] == 1:
                counter_class1_correct = counter_class1_correct + 1
            else: 
                counter_class2_correct = counter_class2_correct + 1
        else:
            if meta_learner_inputs[0][i,0] == 1:
                counter_class1_incorrect = counter_class1_incorrect + 1
            else: 
                counter_class2_incorrect = counter_class2_incorrect + 1
                
    accuracy = accuracy_per_subject.mean(axis=0)
    accuracy_class1 = counter_class1_correct / (counter_class1_correct + counter_class1_incorrect)
    accuracy_class2 = counter_class2_correct / (counter_class2_correct + counter_class2_incorrect)
    balanced_accuracy = (accuracy_class1 + accuracy_class2) / 2                           
     
    feature_importances = np.ones((3))    
    oob_accuracy = 1
    log_loss_value = 1
    fpr, tpr, thresholds = roc_curve(meta_learner_inputs[0][:,0], vote_softmax_raw_2[:])
    mean_fpr = np.linspace(0, 1, 100)
    tprs = np.interp(mean_fpr, fpr, tpr)
    roc_auc = auc(fpr, tpr)
    fraction_positives, mean_predicted_value = calibration_curve(meta_learner_inputs[0][:,0],vote_softmax_raw_2[:], n_bins=10)
                  
    return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value


def meta_learner_2nd_level_RF(numrun):

    global STANDARDPATH, OPTIONS
    
    print(numrun)
    random_state_seed = numrun


    #Import inputs

    meta_learner_inputs = []
    meta_learner_train_inputs = []
    checked_classifiers = []
    for i_classifier in range(len(OPTIONS['names_1stlevel_classifiers'])):
        full_path_cv_option_features = STANDARDPATH + 'metalearner_input' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][i_classifier] + '_save_cv_fold_'+ str (random_state_seed) + '_predictions.txt'
        full_path_cv_option_train = STANDARDPATH + 'metalearner_input' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][i_classifier] + '_save_cv_fold_'+ str (random_state_seed) + '_train_predictions.txt'
        checked_classifiers.append(OPTIONS['names_1stlevel_classifiers'][i_classifier])
        meta_learner_inputs.append(np.array(read_csv(full_path_cv_option_features,sep="\s", header=None, engine='python')))
        meta_learner_train_inputs.append(np.array(read_csv(full_path_cv_option_train,sep="\s", header=None, engine='python')))


    meta_learner_features_train = np.zeros((len(meta_learner_train_inputs[0]), len(checked_classifiers)))
    meta_learner_features_test = np.zeros((len(meta_learner_inputs[0]), len(checked_classifiers)))
    for i_classifier in range(len(checked_classifiers)):
        meta_learner_features_train[:,i_classifier] = meta_learner_train_inputs[i_classifier][:,1]
        meta_learner_features_test[:,i_classifier] = meta_learner_inputs[i_classifier][:,1]

    meta_learner_labels_train = np.zeros(len(meta_learner_train_inputs[0]))
    meta_learner_labels_train = meta_learner_train_inputs[0][:,0]
    meta_learner_labels_test = np.zeros(len(meta_learner_inputs[0]))
    meta_learner_labels_test = meta_learner_inputs[0][:,0]

    
    clf = RandomForestClassifier(n_estimators= 1000, criterion= 'gini', max_depth= None, min_samples_split= 2, min_samples_leaf= 1, bootstrap= True, oob_score=True, random_state=random_state_seed)
    clf = clf.fit(meta_learner_features_train, meta_learner_labels_train)
    
    y_prediction = np.zeros((len(meta_learner_labels_test), 3))
    
    y_prediction[:,0] = clf.predict(meta_learner_features_test)
        
    y_prediction[:,1] = meta_learner_labels_test[:]
    
    
    accuracy, accuracy_class1, accuracy_class2, balanced_accuracy = accuracies_metrics_for_classifiers(y_prediction, meta_learner_labels_test)
    oob_accuracy = 1
    log_loss_value = log_loss(meta_learner_labels_test, clf.predict_proba(meta_learner_features_test), normalize=True)
        
    feature_importances = clf.feature_importances_
    fpr, tpr, thresholds = roc_curve(meta_learner_labels_test, clf.predict_proba(meta_learner_features_test)[:,1])
    mean_fpr = np.linspace(0, 1, 100)
    tprs = np.interp(mean_fpr, fpr, tpr)
    roc_auc = auc(fpr, tpr)
    fraction_positives, mean_predicted_value = calibration_curve(meta_learner_labels_test, clf.predict_proba(meta_learner_features_test)[:,1], n_bins=10)

    return accuracy, accuracy_class1, accuracy_class2, balanced_accuracy, oob_accuracy, log_loss_value, feature_importances, fpr, tpr, tprs, roc_auc, fraction_positives, mean_predicted_value



def integrate_2nd_lvl_results():
    
    global STANDARDPATH, OPTIONS, current_model

    balanced_accuracies = np.zeros((OPTIONS['number_iterations'],len(OPTIONS['abbreviations_features'])))
    load_OPTIONS =  [None]*len(OPTIONS['abbreviations_features'])
    for i_all_classifiers in range(len(OPTIONS['abbreviations_features'])):
        load_OPTIONS[i_all_classifiers] = STANDARDPATH + 'accuracy' + os.sep + OPTIONS['name_model'] + OPTIONS['abbreviations_features'][i_all_classifiers] + '_per_round_balanced_accuracy.txt'
        balanced_accuracies[:,i_all_classifiers] = np.array(np.transpose(read_csv(load_OPTIONS[i_all_classifiers], header=None)))[:,0]

    save_option_prebuild = STANDARDPATH + 'accuracy' + os.sep
    save_option = save_option_prebuild + OPTIONS['name_model'] + '_all_balanced_accs_per_iteration.txt'   
    np.savetxt(save_option, balanced_accuracies, delimiter=',', fmt='%1.3f', header='demo,activation, gppi_phasic, gppi_sustained,graph_gppi_phasic, graph_gppi_sustained, structural, variability, Softmax Voting,2nd level Random Forest', comments='')



def save_current_model(current_model):

    os.makedirs(os.path.dirname(STANDARDPATH + 'metalearner_input' + os.sep ), exist_ok=True)
    save_cv_option_cur_model =  STANDARDPATH + 'metalearner_input' + os.sep + 'current_model.txt'
    with open(save_cv_option_cur_model, 'wb') as AutoPickleFile:
        pickle.dump((current_model), AutoPickleFile)  


if __name__ == '__main__':
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    pool = Pool(12)
    runs_list = []
    outcomes = []
    for i in range (OPTIONS['number_iterations']):
        runs_list.append(i)
    pool.map(prepare_data,runs_list)
    for model in range(len(OPTIONS['abbreviations_features'])):
        save_current_model(model)
        if model < len(OPTIONS['names_1stlevel_classifiers']):
            outcomes[:] = pool.map(classification_pipeline,runs_list)
        elif model == len(OPTIONS['names_1stlevel_classifiers']):
            outcomes[:] = pool.map(meta_learner_softmax_voting,runs_list)
        elif model == len(OPTIONS['names_1stlevel_classifiers'])+1:
            outcomes[:] = pool.map(meta_learner_2nd_level_RF,runs_list)
        accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat, dummy_acc_flat = list_to_flatlist(outcomes)
        save_performance_measures(accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat, dummy_acc_flat)
        aggregate_scores(accuracy_flat, accuracy_class1_flat, accuracy_class2_flat, balanced_accuracy_flat, oob_accuracy_flat, log_loss_value_flat, feature_importances_flat, fpr_flat, tpr_flat, tprs_flat, roc_auc_flat, fraction_positives_flat, mean_predicted_value_flat)
    integrate_2nd_lvl_results()
    pool.close()
    pool.join()
    elapsed_time = time.time() - start_time
    print(elapsed_time)