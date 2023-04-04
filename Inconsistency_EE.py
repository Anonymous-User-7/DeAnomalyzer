import warnings
warnings.filterwarnings('ignore')
import sys
import os
import pandas as pd
import mat73
from scipy.io import loadmat
import numpy as np
from sklearn import metrics
from copy import deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import bisect

import matlab.engine
eng = matlab.engine.start_matlab()

datasetFolderDir = 'Dataset/'

withGT = True

fname = str(sys.argv[1])

datasetFolderDir = 'Dataset/'


def ee(filename, parameters_mat, parameters_sk):
    folderpath = datasetFolderDir
    global withGT
    
    if os.path.exists(folderpath+filename+".mat") == 1:
        try:
            df = loadmat(folderpath+filename+".mat")
        except NotImplementedError:
            df = mat73.loadmat(folderpath+filename+".mat")

        gt=df["y"]
        gt = gt.reshape((len(gt)))
        X=df['X']
        if np.isnan(X).any():
            print("File contains NaN")
            return
    elif os.path.exists(folderpath+filename+".csv") == 1:
        X = pd.read_csv(folderpath+filename+".csv")
        if 'target' in X.columns:
            target=X["target"].to_numpy()
            X=X.drop("target", axis=1)
            gt = target
        else:
            gt = []
            withGT = False
        if X.isna().any().any() == 1:
            print("File contains NaN")
            return
    else:
        print("File doesn't exist")
        return
    
    
    # ## Rearrange "IF" and "LOF"
    mod_parameters_mat = deepcopy(parameters_mat)
    mod_parameters_sk = deepcopy(parameters_sk)
    
    if_cont, lof_cont = IF_LOF_ContFactor(X)
    
    if lof_cont != 0:
        bisect.insort(mod_parameters_sk[3][2], lof_cont)
    if if_cont != 0:
        bisect.insort(mod_parameters_sk[3][2], if_cont)
    
    # ##
    
    mod_parameters_mat[1][1] = 0.1
    
    DefaultARI, DefaultF1_sk, DefaultF1_mat = runEE(filename, X, gt, parameters_sk, parameters_mat)
    
    print("Default settings: ")
    print("\tMean Mutual-ARI: ", DefaultARI)
    if withGT:
        print("\tF1 Score: ")
        print("\t\tMatlab: ", DefaultF1_mat)
        print("\t\tScikit-Learn: ", DefaultF1_sk)
        
    
    blind_route_sk, blind_route_mat, p_s, p_m = get_blind_route(X, gt, filename, deepcopy(mod_parameters_sk), deepcopy(mod_parameters_mat))
    
    UninformedARI = str(blind_route_sk[-1][3][-1][1])
    UninformedF1_sk = str(blind_route_sk[-1][3][-1][2])
    UninformedF1_mat = str(blind_route_mat[-1][3][-1][3])
    
    print("Univariate Search: ")
    print("\tMean Mutual-run ARI: ", UninformedARI)
    if withGT:
        print("\tF1 Score: ")
        print("\t\tMatlab: ", UninformedF1_mat)
        print("\t\tScikit-Learn: ", UninformedF1_sk)
    print("\tOutput Parameters:")
    print("\n\tMatlab: ", end='')
    for i in range(len(p_m)):
            print(p_m[i][0],":", p_m[i][1], end=', ')
    print("\n")
    print("\n\tScikit-Learn: ", end='')
    for i in range(len(p_s)):
            print(p_s[i][0],":", p_s[i][1], end=', ')
    print("\n")
    
    if withGT:
        informed_route_sk, informed_route_mat, p_s, p_m = get_informed_route(X, gt, filename, deepcopy(mod_parameters_sk), deepcopy(mod_parameters_mat))
        InformedARI = str(informed_route_sk[-1][3][-1][1])
        InformedF1_sk = str(informed_route_sk[-1][3][-1][2])
        InformedF1_mat = str(informed_route_mat[-1][3][-1][3])
        print("Bivariate Search: ")
        print("\tMean Mutual-run ARI: ", InformedARI)
        if withGT:
            print("\tF1 Score: ")
            print("\t\tMatlab: ", InformedF1_mat)
            print("\t\tScikit-Learn: ", InformedF1_sk)
        print("\tOutput Parameters:")
        print("\n\tMatlab: ", end='')
        for i in range(len(p_m)):
            print(p_m[i][0],":", p_m[i][1], end=', ')
        print("\n")
        print("\n\tScikit-Learn: ", end='')
        for i in range(len(p_s)):
            print(p_s[i][0],":", p_s[i][1], end=', ')
        print("\n")
    


def IF_LOF_ContFactor(X):
    labels = []
    num_label = 0
    for i in range(5):
        clustering = IsolationForest().fit(X)
    
        l = clustering.predict(X)
        num_label = len(l)
        l = [0 if x == 1 else 1 for x in l]
        labels.append(l)
    _, counts_if = np.unique(labels, return_counts=True)
    if_per = min(counts_if)/(num_label*5)

    
    labels_lof = LocalOutlierFactor().fit_predict(X)
    _, counts_lof = np.unique(labels_lof, return_counts=True)
    lof_per = min(counts_lof)/(num_label)
    
    if if_per == 1:
        if_per = 0
    if lof_per == 1:
        lof_per = 0
    
    return if_per, lof_per


def get_blind_route(X, gt, filename, paramaters_sk_copy,paramaters_mat_copy_pre):
    blind_route_sk = []
    blind_route_mat = []
    route_temp = []
    for p_sk in range(len(paramaters_sk_copy)):
        parameter_route_sk = []
        ari_scores_sk = []
        passing_param_sk = deepcopy(paramaters_sk_copy)

        default_ari, default_f1_sk, default_f1_mat, route_mat = get_blind_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy_pre)
        if default_ari == -1:
            return [], []
        parameter_route_sk.append([passing_param_sk[p_sk][1], default_ari, default_f1_sk, default_f1_mat])
        ari_scores_sk.append(default_ari)
        blind_route_mat += route_mat
        
        i_def = passing_param_sk[p_sk][2].index(passing_param_sk[p_sk][1])
        if i_def+1 == len(passing_param_sk[p_sk][2]):
            i_pv_sk = i_def-1    
        else:
            i_pv_sk = i_def+1
        
        while True:
            if i_pv_sk >= len(paramaters_sk_copy[p_sk][2]):
                break
            if i_pv_sk < 0:
                break

            passing_param_sk[p_sk][1] = paramaters_sk_copy[p_sk][2][i_pv_sk]
            
            ari_score, f1_score_sk, f1_score_mat, route_mat = get_blind_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy_pre)
            route_temp += route_mat
            if ari_score >= np.max(ari_scores_sk):
                parameter_route_sk.append([passing_param_sk[p_sk][1], ari_score, f1_score_sk, f1_score_mat])
                ari_scores_sk.append(ari_score)
                blind_route_mat += route_temp
                route_temp = []
            
            if ari_score != np.max(ari_scores_sk):
                
                if i_pv_sk - 1 > i_def:
                    break
                elif i_pv_sk - 1 == i_def:
                    i_pv_sk = i_def - 1
                else:
                    break
            else:
                if i_pv_sk > i_def:
                    i_pv_sk += 1
                else:
                    i_pv_sk -= 1
        ari_scores_sk = np.array(ari_scores_sk)
        max_index = np.where(ari_scores_sk == max(ari_scores_sk))[0][-1]
        
        default_index = np.where(ari_scores_sk == default_ari)[0][0]
        paramaters_sk_copy[p_sk][1] = parameter_route_sk[max_index][0]
        blind_route_sk.append([paramaters_sk_copy[p_sk][0], max_index, default_index, parameter_route_sk])
    return blind_route_sk, blind_route_mat, paramaters_sk_copy,paramaters_mat_copy_pre
    
    
def get_blind_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy):
    blind_route = []
    
    for p in range(len(paramaters_mat_copy)):
        
        if paramaters_mat_copy[0][1] == "fmcd" and p != 0:
            if p == 4 or p == 5 or p == 6 or p == 7 or p == 8:
                continue
            
        if paramaters_mat_copy[0][1] == "ogk" and p != 0:
            if p == 1 or p == 2 or p == 3 or p==6 or p == 7 or p == 8:
                continue
        
        if paramaters_mat_copy[0][1] == "olivehawkins" and p != 0:
            if p == 3 or p == 4 or p == 5:
                continue
        
        parameter_route = []
        ari_scores = []
        passing_param = deepcopy(paramaters_mat_copy)

        default_ari, default_f1_sk, default_f1_mat = runEE(filename, X, gt, passing_param_sk, passing_param)
        if default_ari == -1:
            return -1, -1, -1, []
        parameter_route.append([passing_param[p][1], default_ari, default_f1_sk, default_f1_mat])
        ari_scores.append(default_ari)
        i_def = passing_param[p][2].index(passing_param[p][1])
        if i_def+1 == len(paramaters_mat_copy[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(paramaters_mat_copy[p][2]):
                break
            if i_pv < 0:
                break

            passing_param[p][1] = paramaters_mat_copy[p][2][i_pv]
            ari_score, f1_score_sk, f1_score_mat = runEE(filename, X, gt, passing_param_sk, passing_param)
            if ari_score >= np.max(ari_scores):
                parameter_route.append([passing_param[p][1], ari_score, f1_score_sk, f1_score_mat])
                ari_scores.append(ari_score)
            if np.max(ari_scores) == 1:
                break
            if ari_score != np.max(ari_scores):
                
                if i_pv - 1 > i_def:
                    break
                elif i_pv - 1 == i_def:
                    i_pv = i_def - 1
                else:
                    break
            else:
                if i_pv > i_def:
                    i_pv += 1
                else:
                    i_pv -= 1
        ari_scores = np.array(ari_scores)
        max_index = np.where(ari_scores == max(ari_scores))[0][-1]
        default_index = np.where(ari_scores == default_ari)[0][0]
        paramaters_mat_copy[p][1] = parameter_route[max_index][0]
        blind_route.append([paramaters_mat_copy[p][0], max_index, default_index, parameter_route])
    return blind_route[-1][3][-1][1], blind_route[-1][3][-1][2], blind_route[-1][3][-1][3], blind_route


def get_informed_route(X, gt, filename, paramaters_sk_copy,paramaters_mat_copy_pre):
    informed_route_sk = []
    informed_route_mat = []
    route_temp = []
    for p_sk in range(len(paramaters_sk_copy)):
        parameter_route_sk = []
        ari_scores_sk = []
        f1_scores_sk = []
        passing_param_sk = deepcopy(paramaters_sk_copy)

        default_ari, default_f1_sk, default_f1_mat, route_mat = get_informed_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy_pre)
        if default_ari == -1:
            return [], []
        parameter_route_sk.append([passing_param_sk[p_sk][1], default_ari, default_f1_sk, default_f1_mat])
        ari_scores_sk.append(default_ari)
        f1_scores_sk.append(default_f1_sk)
        informed_route_mat += route_mat
        
        i_def = passing_param_sk[p_sk][2].index(passing_param_sk[p_sk][1])
        if i_def+1 == len(passing_param_sk[p_sk][2]):
            i_pv_sk = i_def-1    
        else:
            i_pv_sk = i_def+1
        
        while True:
            if i_pv_sk >= len(paramaters_sk_copy[p_sk][2]):
                break
            if i_pv_sk < 0:
                break

            passing_param_sk[p_sk][1] = paramaters_sk_copy[p_sk][2][i_pv_sk]
            
            
            ari_score, f1_score_sk, f1_score_mat, route_mat = get_informed_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy_pre)
            route_temp += route_mat
            if ari_score >= np.max(ari_scores_sk) and f1_score_sk >= np.max(f1_scores_sk):
                parameter_route_sk.append([passing_param_sk[p_sk][1], ari_score, f1_score_sk, f1_score_mat])
                ari_scores_sk.append(ari_score)
                f1_scores_sk.append(f1_score_sk)
                informed_route_mat += route_temp
                route_temp = []
            if ari_score != np.max(ari_scores_sk):
                
                if i_pv_sk - 1 > i_def:
                    break
                elif i_pv_sk - 1 == i_def:
                    i_pv_sk = i_def - 1
                else:
                    break
            else:
                if i_pv_sk > i_def:
                    i_pv_sk += 1
                else:
                    i_pv_sk -= 1
        ari_scores_sk = np.array(ari_scores_sk)
        max_index = np.where(ari_scores_sk == max(ari_scores_sk))[0][-1]
        
        default_index = np.where(ari_scores_sk == default_ari)[0][0]
        paramaters_sk_copy[p_sk][1] = parameter_route_sk[max_index][0]
        informed_route_sk.append([paramaters_sk_copy[p_sk][0], max_index, default_index, parameter_route_sk])

    return informed_route_sk, informed_route_mat, paramaters_sk_copy,paramaters_mat_copy_pre
    
    
def get_informed_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy):
    informed_route = []
    
    for p in range(len(paramaters_mat_copy)):
        
        if paramaters_mat_copy[0][1] == "fmcd" and p != 0:
            if p == 4 or p == 5 or p == 6 or p == 7 or p == 8:
                continue
            
        if paramaters_mat_copy[0][1] == "ogk" and p != 0:
            if p == 1 or p == 2 or p == 3 or p==6 or p == 7 or p == 8:
                continue
        
        if paramaters_mat_copy[0][1] == "olivehawkins" and p != 0:
            if p == 3 or p == 4 or p == 5:
                continue
        
        parameter_route = []
        ari_scores = []
        f1_scores = []
        passing_param = deepcopy(paramaters_mat_copy)

        default_ari, default_f1_sk, default_f1_mat = runEE(filename, X, gt, passing_param_sk, passing_param)
        if default_ari == -1:
            return -1, -1, -1, []
        parameter_route.append([passing_param[p][1], default_ari, default_f1_sk, default_f1_mat])
        ari_scores.append(default_ari)
        f1_scores.append(default_f1_mat)
        i_def = passing_param[p][2].index(passing_param[p][1])
        if i_def+1 == len(paramaters_mat_copy[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(paramaters_mat_copy[p][2]):
                break
            if i_pv < 0:
                break

            passing_param[p][1] = paramaters_mat_copy[p][2][i_pv]
            ari_score, f1_score_sk, f1_score_mat = runEE(filename, X, gt, passing_param_sk, passing_param)
            if ari_score >= np.max(ari_scores) and f1_score_mat >= np.max(f1_scores):
                parameter_route.append([passing_param[p][1], ari_score, f1_score_sk, f1_score_mat])
                ari_scores.append(ari_score)
                f1_scores.append(f1_score_mat)
            if np.max(ari_scores) == 1:
                break
            if ari_score != np.max(ari_scores):
                
                if i_pv - 1 > i_def:
                    break
                elif i_pv - 1 == i_def:
                    i_pv = i_def - 1
                else:
                    break
            else:
                if i_pv > i_def:
                    i_pv += 1
                else:
                    i_pv -= 1
        ari_scores = np.array(ari_scores)
        max_index = np.where(ari_scores == max(ari_scores))[0][-1]
        default_index = np.where(ari_scores == default_ari)[0][0]
        paramaters_mat_copy[p][1] = parameter_route[max_index][0]
        informed_route.append([paramaters_mat_copy[p][0], max_index, default_index, parameter_route])
    return informed_route[-1][3][-1][1], informed_route[-1][3][-1][2], informed_route[-1][3][-1][3], informed_route


def runEE(filename, X, gt, param_sk, param_mat):
    labelFile_sk = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1])
    labelFile_mat = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1]) + "_" + str(param_mat[3][1]) + "_" + str(param_mat[4][1]) + "_" + str(param_mat[5][1]) + "_" + str(param_mat[6][1]) + "_" + str(param_mat[7][1]) + "_" + str(param_mat[8][1])

    if os.path.exists("Labels/EE_Sk/Labels_Sk_EE_"+labelFile_sk+".csv") == 0:
        skf1 = get_sk_f1(filename, param_sk, X, gt)
        if skf1 == -1:
            return -1, -1, -1
        
    if os.path.exists("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile_mat+".csv") == 0:        
        frr=open("GD_ReRun/MatEE.csv", "a")
        frr.write(filename+","+str(param_mat[0][1])+","+str(param_mat[1][1])+","+str(param_mat[2][1])+","+str(param_mat[3][1])+","+str(param_mat[4][1])+","+str(param_mat[5][1])+","+str(param_mat[6][1])+","+str(param_mat[7][1])+","+str(param_mat[8][1])+'\n')
        frr.close()
        try:
            eng.MatEE_Rerun(nargout=0)
            frr=open("GD_ReRun/MatEE.csv", "w")
            frr.write('Filename,Method,OutlierFraction,NumTrials,BiasCorrection,NumOGKIterations,UnivariateEstimator,ReweightingMethod,NumConcentrationSteps,StartMethod\n')
            frr.close()
            if os.path.exists("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile_mat+".csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0)  
            

    labels_sk =  pd.read_csv("Labels/EE_Sk/Labels_Sk_EE_"+labelFile_sk+".csv", header=None).to_numpy()
    labels_mat =  pd.read_csv("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile_mat+".csv", header=None).to_numpy()
    
    ari = []
    
    for i in range(len(labels_sk)):
        for j in range(len(labels_mat)):
            ari.append(adjusted_rand_score(labels_sk[i], labels_mat[j]))
    if withGT:
        mat_f1 = []
        for i in range(10):
            mat_f1.append(metrics.f1_score(gt, labels_mat[i]))
        sk_f1 = []
        for i in range(10):
            sk_f1.append(metrics.f1_score(gt, labels_sk[i]))
        
        return np.mean(ari), np.mean(sk_f1), np.mean(mat_f1)
    else:
        return np.mean(ari), -1, -1
    
def get_sk_f1(filename, param_sk, X, gt):
    labelFile = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1])
  
    sp = param_sk[0][1]
    ac = param_sk[1][1]
    sf = param_sk[2][1]
    cont = param_sk[3][1]
    
    labels = []
    f1 = []
    for i in range(10):
        clustering = EllipticEnvelope(store_precision=sp, assume_centered=ac, 
                                     support_fraction=sf, contamination=cont).fit(X)
    
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels.append(l)
        if withGT:
            f1.append(metrics.f1_score(gt, l))
        
    if os.path.exists("Labels/EE_Sk/Labels_Sk_EE_"+labelFile+".csv") == 0:
        fileLabels=open("Labels/EE_Sk/Labels_Sk_EE_"+labelFile+".csv", 'a')
        for l in labels:
            fileLabels.write(','.join(str(s) for s in l) + '\n')
        fileLabels.close()
    
    if withGT:
        return np.mean(f1)
    else:
        return -1
    
if __name__ == '__main__':
    print("\nRunning DeAnomalyzer on ", fname)
    folderpath = datasetFolderDir
    
    parameters_mat = []

    Method = ["olivehawkins", "fmcd", "ogk"];
    OutlierFraction = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5];
    NumTrials = [2, 5, 10, 25, 50, 100, 250, 500, 750, 1000];
    BiasCorrection = [1, 0];
    NumOGKIterations = [1, 2, 3];
    UnivariateEstimator = ["tauscale", "qn"];
    ReweightingMethod = ["rfch", "rmvn"];
    NumConcentrationSteps = [2, 5, 10, 15, 20];
    StartMethod = ["elemental","classical", "medianball"];    
    
    parameters_mat.append(["Method", "fmcd", Method])
    parameters_mat.append(["OutlierFraction", 0.5, OutlierFraction])
    parameters_mat.append(["NumTrials", 500, NumTrials])
    parameters_mat.append(["BiasCorrection", 1, BiasCorrection])
    parameters_mat.append(["NumOGKIterations", 2, NumOGKIterations])
    parameters_mat.append(["UnivariateEstimator", "tauscale", UnivariateEstimator])
    parameters_mat.append(["ReweightingMethod", "rfch", ReweightingMethod])
    parameters_mat.append(["NumConcentrationSteps", 10, NumConcentrationSteps])
    parameters_mat.append(["StartMethod", "classical", StartMethod])
    
    
    parameters_sk = []
    store_precision = [True, False]
    assume_centered = [True, False]
    support_fraction = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    contamination = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    parameters_sk.append(["store_precision", True, store_precision])
    parameters_sk.append(["assume_centered", False, assume_centered])
    parameters_sk.append(["support_fraction", None, support_fraction])
    parameters_sk.append(["contamination", 0.1, contamination])
    
    
    frr=open("GD_ReRun/MatEE.csv", "w")
    frr.write('Filename,Method,OutlierFraction,NumTrials,BiasCorrection,NumOGKIterations,UnivariateEstimator,ReweightingMethod,NumConcentrationSteps,StartMethod\n')
    frr.close()
    
    if ".csv" in fname:
        fname = fname.split(".csv")[0]
        
    ee(fname, parameters_mat, parameters_sk)
    
    eng.quit()