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
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import bisect 

import subprocess
import matlab.engine
eng = matlab.engine.start_matlab()

datasetFolderDir = 'Dataset/'


withGT = True

fname = str(sys.argv[1])


def ocsvm(filename, parameters_r, parameters_mat, parameters_sk):
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
    
    ## Rearrange "IF" and "LOF"
    mod_parameters_r = deepcopy(parameters_r)
    mod_parameters_mat = deepcopy(parameters_mat)
    mod_parameters_sk = deepcopy(parameters_sk)
    
    if_cont, lof_cont = IF_LOF_ContFactor(X)
    if lof_cont != 0:
        bisect.insort(mod_parameters_mat[0][2], lof_cont)
        bisect.insort(mod_parameters_sk[5][2], lof_cont)
        bisect.insort(mod_parameters_r[5][2], lof_cont)
    
    if if_cont != 0:
        bisect.insort(mod_parameters_mat[0][2], if_cont)
        bisect.insort(mod_parameters_sk[5][2], if_cont)
        bisect.insort(mod_parameters_r[5][2], if_cont)
    
    # ##
    
    # #Shortcut
    # mod_parameters_r[5][1] = "IF"
    # mod_parameters_r[2][1] = "auto"
    
    # mod_parameters_mat[0][1] = "IF"
    # mod_parameters_mat[1][1] = "auto"
    
    # mod_parameters_sk[5][1] = "IF"

    # print(mod_parameters_mat)
    
    
    tools = ['R', 'Matlab', 'Sklearn']
    
    DefaultARI, DefaultF1_r, DefaultF1_mat, DefaultF1_sk = runOCSVM(filename, X, gt, parameters_r, parameters_mat, parameters_sk)

    print("Default settings: ")
    print("\tMean Mutual-ARI: ", DefaultARI)
    if withGT:
        print("\tF1 Score: ")
        print("\t\tR: ", DefaultF1_r)
        print("\t\tMatlab: ", DefaultF1_mat)
        print("\t\tScikit-Learn: ", DefaultF1_sk)
    
    param_r_b, param_mat_b, param_sk_b = deepcopy(mod_parameters_r), deepcopy(mod_parameters_mat), deepcopy(mod_parameters_sk)
    ari_score, f1_score_r, f1_score_mat, f1_score_sk, p_r, p_m, p_s = get_blind_route(X, gt, filename, param_r_b, param_mat_b, param_sk_b, tools)
    
    UninformedARI = str(ari_score)
    UninformedF1_r = str(f1_score_r)
    UninformedF1_mat = str(f1_score_mat)
    UninformedF1_sk = str(f1_score_sk)
    
    print("Univariate Search: ")
    print("\tMean Mutual-run ARI: ", UninformedARI)
    if withGT:
        print("\tF1 Score: ")
        print("\t\tR: ", UninformedF1_r)
        print("\t\tMatlab: ", UninformedF1_mat)
        print("\t\tScikit-Learn: ", UninformedF1_sk)
    print("\tOutput Parameters:")
    print("\n\tR: ", end='')
    for i in range(len(p_r)):
        print(p_r[i][0],":", p_r[i][1], end=', ')
    print("\n")
    print("\n\tMatlab: ", end='')
    for i in range(len(p_m)):
        print(p_m[i][0],":", p_m[i][1], end=', ')
    print("\n")
    print("\n\tScikit-Learn: ", end='')
    for i in range(len(p_s)):
        print(p_s[i][0],":", p_s[i][1], end=', ')
    print("\n")
    
    
    if withGT:
        param_r_u, param_mat_u, param_sk_u = deepcopy(mod_parameters_r), deepcopy(mod_parameters_mat), deepcopy(mod_parameters_sk)
        ari_score, f1_score_r, f1_score_mat, f1_score_sk, p_r_g, p_m_g, p_s_g = get_guided_route(X, gt, filename, param_r_u, param_mat_u, param_sk_u, tools)    
        
        
        InformedARI = str(ari_score)
        InformedF1_r = str(f1_score_r)
        InformedF1_mat = str(f1_score_mat)
        InformedF1_sk = str(f1_score_sk)
        print("Bivariate Search: ")
        print("\tMean Mutual-run ARI: ", InformedARI)
        if withGT:
            print("\tF1 Score: ")
            print("\t\tR: ", InformedF1_r)
            print("\t\tMatlab: ", InformedF1_mat)
            print("\t\tScikit-Learn: ", InformedF1_sk)
        print("\tOutput Parameters:")
        print("\n\tR: ", end='')
        for i in range(len(p_r_g)):
            print(p_r_g[i][0],":", p_r_g[i][1], end=', ')
        print("\n")
        print("\n\tMatlab: ", end='')
        for i in range(len(p_m_g)):
            print(p_m_g[i][0],":", p_m_g[i][1], end=', ')
        print("\n")
        print("\n\tScikit-Learn: ", end='')
        for i in range(len(p_s_g)):
            print(p_s_g[i][0],":", p_s_g[i][1], end=', ')
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

def get_blind_route(X, gt, filename, parameters_r_copy,parameters_mat_copy, parameters_sk_copy, tools):
    if tools == []:
        ari_score, f1_score_r, f1_score_mat, f1_score_sk = runOCSVM(filename, X, gt, parameters_r_copy, parameters_mat_copy, parameters_sk_copy)
        return ari_score, f1_score_r, f1_score_mat, f1_score_sk, parameters_r_copy, parameters_mat_copy, parameters_sk_copy

    current_tool = tools[0]
    
    # print(current_tool, end=' - ')
    
    parameters_copy = []
    if current_tool == 'Sklearn':
        parameters_copy = parameters_sk_copy
    elif current_tool == 'Matlab':
        parameters_copy = parameters_mat_copy
    else:
        parameters_copy = parameters_r_copy
            
    
    for p_c in range(len(parameters_copy)):
        parameter_route_c = []
        ari_scores_c = []
        passing_param_c = parameters_copy

        default_ari, default_f1_r, default_f1_mat, default_f1_sk, _, _, _ = get_blind_route(X, gt, filename, parameters_r_copy, parameters_mat_copy, parameters_sk_copy, tools[1:])
        if default_ari == -1:
            return -1, -1, -1, -1, [], [], []
        parameter_route_c.append([passing_param_c[p_c][1], default_ari, default_f1_r, default_f1_mat, default_f1_sk])
        ari_scores_c.append(default_ari)

        i_def = passing_param_c[p_c][2].index(passing_param_c[p_c][1])
        if i_def+1 == len(passing_param_c[p_c][2]):
            i_pv_c = i_def-1
        else:
            i_pv_c = i_def+1
        
        while True:
            if i_pv_c >= len(parameters_copy[p_c][2]):
                break
            if i_pv_c < 0:
                break

            passing_param_c[p_c][1] = parameters_copy[p_c][2][i_pv_c]
            
            ari_score, f1_score_r, f1_score_mat, f1_score_sk, _, _, _ = get_blind_route(X, gt, filename, parameters_r_copy, parameters_mat_copy, parameters_sk_copy, tools[1:])
            if ari_score >= np.max(ari_scores_c):
                parameter_route_c.append([passing_param_c[p_c][1], ari_score, f1_score_r, f1_score_mat, f1_score_sk])
                ari_scores_c.append(ari_score)
            if ari_score != np.max(ari_scores_c):
                
                if i_pv_c - 1 > i_def:
                    break
                elif i_pv_c - 1 == i_def:
                    i_pv_c = i_def - 1
                else:
                    break
            else:
                if i_pv_c > i_def:
                    i_pv_c += 1
                else:
                    i_pv_c -= 1
        ari_scores_c = np.array(ari_scores_c)
        max_index = np.where(ari_scores_c == max(ari_scores_c))[0][-1]
        parameters_copy[p_c][1] = parameter_route_c[max_index][0]
    return parameter_route_c[-1][1], parameter_route_c[-1][2], parameter_route_c[-1][3], parameter_route_c[-1][4], parameters_r_copy, parameters_mat_copy, parameters_sk_copy

def get_guided_route(X, gt, filename, parameters_r_copy,parameters_mat_copy, parameters_sk_copy, tools):
    if tools == []:
        ari_score, f1_score_r, f1_score_mat, f1_score_sk = runOCSVM(filename, X, gt, parameters_r_copy, parameters_mat_copy, parameters_sk_copy)
        return ari_score, f1_score_r, f1_score_mat, f1_score_sk, parameters_r_copy, parameters_mat_copy, parameters_sk_copy

    current_tool = tools[0]
    
    # print(current_tool, end=' - ')
    
    parameters_copy = []
    if current_tool == 'Sklearn':
        parameters_copy = parameters_sk_copy
    elif current_tool == 'Matlab':
        parameters_copy = parameters_mat_copy
    else:
        parameters_copy = parameters_r_copy
            
    
    for p_c in range(len(parameters_copy)):
        parameter_route_c = []
        ari_scores_c = []
        f1_scores_c = []
        passing_param_c = parameters_copy

        default_ari, default_f1_r, default_f1_mat, default_f1_sk, _, _, _ = get_blind_route(X, gt, filename, parameters_r_copy, parameters_mat_copy, parameters_sk_copy, tools[1:])
        if default_ari == -1:
            return -1, -1, -1, -1, [], [], []
        parameter_route_c.append([passing_param_c[p_c][1], default_ari, default_f1_r, default_f1_mat, default_f1_sk])
        ari_scores_c.append(default_ari)
        f1_scores_c.append((default_f1_r+ default_f1_mat+ default_f1_sk)/3)
        
        i_def = passing_param_c[p_c][2].index(passing_param_c[p_c][1])
        if i_def+1 == len(passing_param_c[p_c][2]):
            i_pv_c = i_def-1
        else:
            i_pv_c = i_def+1
        
        while True:
            if i_pv_c >= len(parameters_copy[p_c][2]):
                break
            if i_pv_c < 0:
                break

            passing_param_c[p_c][1] = parameters_copy[p_c][2][i_pv_c]
            
            ari_score, f1_score_r, f1_score_mat, f1_score_sk, _, _, _ = get_guided_route(X, gt, filename, parameters_r_copy, parameters_mat_copy, parameters_sk_copy, tools[1:])
            if ari_score >= np.max(ari_scores_c) and f1_score_r > np.max(f1_scores_c):
                parameter_route_c.append([passing_param_c[p_c][1], ari_score, f1_score_r, f1_score_mat, f1_score_sk])
                ari_scores_c.append(ari_score)
                f1_scores_c.append((f1_score_r+ f1_score_mat+ f1_score_sk)/3)

            if ari_score != np.max(ari_scores_c):
                
                if i_pv_c - 1 > i_def:
                    break
                elif i_pv_c - 1 == i_def:
                    i_pv_c = i_def - 1
                else:
                    break
            else:
                if i_pv_c > i_def:
                    i_pv_c += 1
                else:
                    i_pv_c -= 1
        ari_scores_c = np.array(ari_scores_c)
        max_index = np.where(ari_scores_c == max(ari_scores_c))[0][-1]
        parameters_copy[p_c][1] = parameter_route_c[max_index][0]
    return parameter_route_c[-1][1], parameter_route_c[-1][2], parameter_route_c[-1][3], parameter_route_c[-1][4], parameters_r_copy, parameters_mat_copy, parameters_sk_copy


def runOCSVM(filename, X, gt, param_r, param_mat, param_sk):
    global withGT
    
    labelFile_r = filename + "_" + str(param_r[0][1]) + "_" + str(param_r[1][1]) + "_" + str(param_r[2][1]) + "_" + str(param_r[3][1]) + "_" + str(param_r[4][1]) + "_" + str(param_r[5][1]) + "_" + str(param_r[6][1]) + "_" + str(param_r[7][1]) + "_" + str(param_r[8][1])
    labelFile_mat = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1]) + "_" + str(param_mat[3][1]) + "_" + str(param_mat[4][1]) + "_" + str(param_mat[5][1]) + "_" + str(param_mat[6][1]) + "_" + str(param_mat[7][1])
    labelFile_sk = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1]) + "_" + str(param_sk[4][1]) + "_" + str(param_sk[5][1]) + "_" + str(param_sk[6][1]) + "_" + str(param_sk[7][1]) + "_" + str(param_sk[8][1])
    
    
    if os.path.exists("Labels/OCSVM_R/"+labelFile_r+".csv") == 0:
        frr=open("GD_ReRun/ROCSVM.csv", "a")
        frr.write(filename+","+str(param_r[0][1])+","+str(param_r[1][1])+","+str(param_r[2][1])+","+str(param_r[3][1])+","+str(param_r[4][1])+","+str(param_r[5][1])+","+str(param_r[6][1])+","+str(param_r[7][1])+","+str(param_r[8][1])+'\n')
        frr.close()
        try:
            subprocess.call((["/usr/local/bin/Rscript", "--vanilla", "ROCSVM_Rerun.r"]))
            frr=open("GD_ReRun/ROCSVM.csv", "w")
            frr.write('Filename,kernel,degree,gamma,coef0,tolerance,nu,shrinking,cachesize,epsilon\n')
            frr.close()
            if os.path.exists("Labels/OCSVM_R/"+labelFile_r+".csv") == 0:      
                print("\nFaild to run Rscript from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Rscript from Python.\n")
            exit(0)  
    if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat+".csv") == 0:        
        frr=open("GD_ReRun/MatOCSVM.csv", "a")
        frr.write(filename+","+str(param_mat[0][1])+","+str(param_mat[1][1])+","+str(param_mat[2][1])+","+str(param_mat[3][1])+","+str(param_mat[4][1])+","+str(param_mat[5][1])+","+str(param_mat[6][1])+","+str(param_mat[7][1])+'\n')
        frr.close()
        try:
            eng.MatOCSVM_Rerun(nargout=0)
            frr=open("GD_ReRun/MatOCSVM.csv", "w")
            frr.write('Filename,ContaminationFraction,KernelScale,Lambda,NumExpansionDimensions,StandardizeData,BetaTolerance,BetaTolerance,GradientTolerance,IterationLimit\n')
            frr.close()
            if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat+".csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0)  
    if os.path.exists("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile_sk+".csv") == 0:
        skf1 = get_sk_f1(filename, param_sk, X, gt)
        if os.path.exists("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile_sk+".csv") == 0:
            if skf1 == -2:
                return -1, -1, -1, -1
            print("\nError running Scikit-learn.\n")
            exit(0) 
    
    labels_r = pd.read_csv("Labels/OCSVM_R/"+labelFile_r+".csv").to_numpy()
    labels_mat = pd.read_csv("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat+".csv", header=None).to_numpy()
    labels_sk = pd.read_csv("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile_sk+".csv", header=None).to_numpy()
    
    
        
        
    ari_mvr = []
    for i in range(len(labels_r)):
        for j in range(len(labels_mat)):
            ari_mvr.append(adjusted_rand_score(np.int64((labels_r[i][1:])*1), labels_mat[j]))
    ari_mvr = np.mean(ari_mvr)

    ari_rvs = []
    for i in range(len(labels_r)):
        for j in range(len(labels_sk)):
            ari_rvs.append(adjusted_rand_score(np.int64((labels_r[i][1:])*1), labels_sk[j]))
    ari_rvs = np.mean(ari_rvs)
    
    ari_mvs = []
    for i in range(len(labels_sk)):
        for j in range(len(labels_mat)):
            ari_mvs.append(adjusted_rand_score(labels_sk[i], labels_mat[j]))
    ari_mvs = np.mean(ari_mvs) 
    
    ari = []  
    ari = (ari_mvr + ari_rvs + ari_mvs)/3
    
    

    if withGT:
        rf1 = metrics.f1_score(gt, np.int64((labels_r[0][1:])*1))
        
        mat_f1 = []
        for i in range(10):
            mat_f1.append(metrics.f1_score(gt, labels_mat[i]))
        matf1 = np.mean(mat_f1)
        
        skf1 = metrics.f1_score(gt, labels_sk[0])
        return ari, rf1, matf1, skf1
    else:
        return ari, -1, -1, -1

def get_sk_f1(filename, param_sk, X, gt):
    labelFile = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1]) + "_" + str(param_sk[4][1]) + "_" + str(param_sk[5][1]) + "_" + str(param_sk[6][1]) + "_" + str(param_sk[7][1]) + "_" + str(param_sk[8][1])
  
    labels = []
    f1 = []
    try:
        clustering = OneClassSVM(kernel=param_sk[0][1], degree=param_sk[1][1], gamma=param_sk[2][1], coef0=param_sk[3][1], tol=param_sk[4][1], nu=param_sk[5][1], 
                              shrinking=param_sk[6][1], cache_size=param_sk[7][1], max_iter=param_sk[8][1]).fit(X)
    except:
        # print("Error running Scikit-learn with current settings.")
        return -2
    l = clustering.predict(X)
    l = [0 if x == 1 else 1 for x in l]
    labels.append(l)
    if withGT:
        f1.append(metrics.f1_score(gt, l))
         
    if os.path.exists("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile+".csv") == 0:
        fileLabels=open("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile+".csv", 'a')
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
    parameters_r = []
    
    kernel = ['linear', 'polynomial', 'radial', 'sigmoid']
    degree = [3, 4, 5, 6]
    gamma = ['scale', 'auto']
    coef0 = [0, 0.1, 0.2, 0.3, 0.4]
    tolerance = [0.1, 0.01, 0.001, 0.0001]
    nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    shrinking = ["TRUE", "FALSE"]
    cachesize = [50, 100, 200, 400]
    epsilon = [0.1, 0.2, 0.01, 0.05]
    
    parameters_r.append(["kernel", 'radial', kernel])
    parameters_r.append(["degree", 3, degree])
    parameters_r.append(["gamma","scale",gamma])
    parameters_r.append(["coef0", 0, coef0])
    parameters_r.append(["tolerance", 0.001, tolerance])
    parameters_r.append(["nu", 0.5, nu])
    parameters_r.append(["shrinking", "TRUE", shrinking])
    parameters_r.append(["cachesize", 200, cachesize])
    parameters_r.append(["epsilon",0.1,epsilon])
    
    
    parameters_mat = []
    
    ContaminationFraction = [0.05, 0.1, 0.15, 0.2, 0.25]
    KernelScale = [1, "auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    Lambda = ["auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    NumExpansionDimensions = ["auto", 2^12, 2^15, 2^17, 2^19]
    StandardizeData = [0, 1]
    BetaTolerance = [1e-2, 1e-3, 1e-4, 1e-5]
    GradientTolerance = [1e-3, 1e-4, 1e-5, 1e-6]
    IterationLimit = [100, 200, 500, 1000, 2000]
    
    parameters_mat.append(["ContaminationFraction", 0.1, ContaminationFraction])
    parameters_mat.append(["KernelScale", 1, KernelScale])
    parameters_mat.append(["Lambda", 'auto', Lambda])
    parameters_mat.append(["NumExpansionDimensions", 'auto', NumExpansionDimensions])
    parameters_mat.append(["StandardizeData", 0, StandardizeData])
    parameters_mat.append(["BetaTolerance", 0.0001, BetaTolerance])
    parameters_mat.append(["GradientTolerance", 0.0001, GradientTolerance])
    parameters_mat.append(["IterationLimit", 1000, IterationLimit])
    
    
    parameters_sk = []
    kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    degree = [3, 4, 5, 6] # Kernel poly only
    gamma = ['scale', 'auto'] # Kernel ‘rbf’, ‘poly’ and ‘sigmoid’
    coef0 = [0.0, 0.1, 0.2, 0.3, 0.4] # Kernel ‘poly’ and ‘sigmoid’
    tol = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    shrinking = [True, False]
    cache_size = [50, 100, 200, 400]
    max_iter = [50, 100, 150, 200, 250, 300, -1]
    
    parameters_sk.append(["kernel", 'rbf', kernel])
    parameters_sk.append(["degree", 3, degree])
    parameters_sk.append(["gamma", 'scale', gamma])
    parameters_sk.append(["coef0", 0.0, coef0])
    parameters_sk.append(["tol", 0.001, tol])
    parameters_sk.append(["nu", 0.5, nu])
    parameters_sk.append(["shrinking", True, shrinking])
    parameters_sk.append(["cache_size", 200, cache_size])
    parameters_sk.append(["max_iter", -1, max_iter])
    
    
    frr=open("GD_ReRun/ROCSVM.csv", "w")
    frr.write('Filename,kernel,degree,gamma,coef0,tolerance,nu,shrinking,cachesize,epsilon\n')
    frr.close()
    
    frr=open("GD_ReRun/MatOCSVM.csv", "w")
    frr.write('Filename,ContaminationFraction,KernelScale,Lambda,NumExpansionDimensions,StandardizeData,BetaTolerance,BetaTolerance,GradientTolerance,IterationLimit\n')
    frr.close()
    
    
    if ".csv" in fname:
        fname = fname.split(".csv")[0]
    ocsvm(fname, parameters_r, parameters_mat, parameters_sk)
    eng.quit()
    