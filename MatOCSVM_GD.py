import warnings
warnings.filterwarnings("ignore")
import sys
import os
import pandas as pd
import mat73
from scipy.io import loadmat
import numpy as np
from sklearn import metrics
from copy import deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
import bisect
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matlab.engine
eng = matlab.engine.start_matlab()

datasetFolderDir = 'Dataset/'

withGT = True

fname = str(sys.argv[1])



def ocsvm(filename, parameters, parameter_iteration):
    folderpath = datasetFolderDir
    parameters_this_file = deepcopy(parameters)
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
    
    # ## Rearrange "IF" and "LOF" on index 0 and "auto" in index 2
    if_cont, lof_cont = IF_LOF_ContFactor(X)
    mod_parameters = deepcopy(parameters)
    
    if lof_cont != 0:
        bisect.insort(mod_parameters[0][2], lof_cont)
    if if_cont != 0:
        bisect.insort(mod_parameters[0][2], if_cont)
    # ##
    
    blind_route = get_blind_route(X, gt, filename, deepcopy(mod_parameters), parameter_iteration)
    
    DefaultARI = str(blind_route[0][3][0][1])
    DefaultF1 = str(blind_route[0][3][0][2])
    print("Default settings: ")
    print("\tCross-run ARI: ", DefaultARI)
    if withGT:
        print("\tF1 Score: ", DefaultF1)
    
    UninformedARI = str(blind_route[-1][3][-1][1])
    UninformedF1 = str(blind_route[-1][3][-1][2])
    print("Univariate Search: ")
    print("\tCross-run ARI: ", UninformedARI)
    if withGT:
        print("\tF1 Score: ", UninformedF1)
    print("\tOutput Parameters:")
    print("\n\t", end='')
    for i in range(len(blind_route)):
        print(blind_route[i][0],":", blind_route[i][3][blind_route[i][1]][0], end=', ')
    print("\n")
    
    if withGT:
        guided_route = get_guided_route(X, gt, filename, deepcopy(mod_parameters), parameter_iteration)
    
        InformedARI = str(guided_route[-1][3][-1][1])
        InformedF1 = str(guided_route[-1][3][-1][2])
        print("Bivariate Search: ")
        print("\tCross-run ARI: ", InformedARI)
        print("\tF1 Score: ", InformedF1)
        print("\tOutput Parameters:")
        print("\n\t", end='')
        for i in range(len(guided_route)):
            print(guided_route[i][0],":", guided_route[i][3][guided_route[i][1]][0], end=', ')
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
    
    
def get_blind_route(X, gt, filename, parameters_this_file, parameter_iteration):
    blind_route = []
    
    for p_i in range(len(parameters_this_file)):
        p = p_i
        
        parameter_route = []
        ari_scores = []
        passing_param = deepcopy(parameters_this_file)

        default_f1, default_ari = runOCSVM(filename, X, gt, passing_param, parameter_iteration)

        parameter_route.append([passing_param[p][1], default_ari, default_f1])
        ari_scores.append(default_ari)
        
        i_def = passing_param[p][2].index(passing_param[p][1])
        if i_def+1 == len(parameters_this_file[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(parameters_this_file[p][2]):
                break
            if i_pv < 0:
                break

            passing_param[p][1] = parameters_this_file[p][2][i_pv]
            f1_score, ari_score = runOCSVM(filename, X, gt, passing_param, parameter_iteration)

            if ari_score > np.max(ari_scores):
                parameter_route.append([passing_param[p][1], ari_score, f1_score])
                ari_scores.append(ari_score)
            
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
        
        max_index = ari_scores.index(max(ari_scores))
        default_index = ari_scores.index(default_ari)
        parameters_this_file[p][1] = parameter_route[max_index][0]
        blind_route.append([parameters_this_file[p][0], max_index, default_index, parameter_route])
    return blind_route
    
def get_guided_route(X, gt, filename, parameters_this_file, parameter_iteration):
    guided_route = []
    
    for p_i in range(len(parameters_this_file)):
        p = p_i
        
        parameter_route = []
        ari_scores = []
        f1_scores = []
        passing_param = deepcopy(parameters_this_file)

        default_f1, default_ari = runOCSVM(filename, X, gt, passing_param, parameter_iteration)

        parameter_route.append([passing_param[p][1], default_ari, default_f1])
        ari_scores.append(default_ari)
        f1_scores.append(default_f1)

        i_def = passing_param[p][2].index(passing_param[p][1])
        if i_def+1 == len(parameters_this_file[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(parameters_this_file[p][2]):
                break
            if i_pv < 0:
                break

            passing_param[p][1] = parameters_this_file[p][2][i_pv]
            f1_score, ari_score = runOCSVM(filename, X, gt, passing_param, parameter_iteration)

            if ari_score > np.max(ari_scores) and f1_score > np.max(f1_scores):
                parameter_route.append([passing_param[p][1], ari_score, f1_score])
                ari_scores.append(ari_score)
                f1_scores.append(f1_score)            
            if ari_score != np.max(ari_scores) and f1_score != np.max(f1_scores):
                
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
        max_index = ari_scores.index(max(ari_scores))
        default_index = ari_scores.index(default_ari)
        parameters_this_file[p][1] = parameter_route[max_index][0]
        guided_route.append([parameters_this_file[p][0], max_index, default_index, parameter_route])
    return guided_route


  

def runOCSVM(filename, X, gt, params, parameter_iteration):
    global withGT
    labelFile = filename + "_" + str(params[0][1]) + "_" + str(params[1][1]) + "_" + str(params[2][1]) + "_" + str(params[3][1]) + "_" + str(params[4][1]) + "_" + str(params[5][1]) + "_" + str(params[6][1]) + "_" + str(params[7][1])

    if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv") == 0:
        frr=open("GD_ReRun/MatOCSVM.csv", "a")
        frr.write(filename+","+str(params[0][1])+","+str(params[1][1])+","+str(params[2][1])+","+str(params[3][1])+","+str(params[4][1])+","+str(params[5][1])+","+str(params[6][1])+","+str(params[7][1])+'\n')
        frr.close()
        try:
            eng.MatOCSVM_Rerun(nargout=0)
            frr=open("GD_ReRun/MatOCSVM.csv", "w")
            frr.write('Filename,ContaminationFraction,KernelScale,Lambda,NumExpansionDimensions,StandardizeData,BetaTolerance,BetaTolerance,GradientTolerance,IterationLimit\n')
            frr.close()
            if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0)    
    f1 = []
    ari = []
    
    labels =  pd.read_csv("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv", header=None).to_numpy()
    if withGT:
        for i in range(10):
            f1.append(metrics.f1_score(gt, labels[i]))
        
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          ari.append(adjusted_rand_score(labels[i], labels[j]))
          
    if withGT:
        return np.mean(f1), np.mean(ari)
    else:
        return -1, np.mean(ari) 
  
    
    
if __name__ == '__main__':
    print("\nRunning DeAnomalyzer on ", fname)
    folderpath = datasetFolderDir
    
    parameters = []

    ContaminationFraction = [0.05, 0.1, 0.15, 0.2, 0.25];
    KernelScale = [1, "auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
    Lambda = ["auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
    NumExpansionDimensions = ["auto", 2^12, 2^15, 2^17, 2^19];
    StandardizeData = [0, 1];
    BetaTolerance = [1e-2, 1e-3, 1e-4, 1e-5];
    GradientTolerance = [1e-3, 1e-4, 1e-5, 1e-6];
    IterationLimit = [100, 200, 500, 1000, 2000];
    
    parameters.append(["ContaminationFraction", 0.1, ContaminationFraction])
    parameters.append(["KernelScale", 1, KernelScale])
    parameters.append(["Lambda", 'auto', Lambda])
    parameters.append(["NumExpansionDimensions", 'auto', NumExpansionDimensions])
    parameters.append(["StandardizeData", 0, StandardizeData])
    parameters.append(["BetaTolerance", 1e-4, BetaTolerance])
    parameters.append(["GradientTolerance", 1e-4, GradientTolerance])
    parameters.append(["IterationLimit", 1000, IterationLimit])
    
    
    frr=open("GD_ReRun/MatOCSVM.csv", "w")
    frr.write('Filename,ContaminationFraction,KernelScale,Lambda,NumExpansionDimensions,StandardizeData,BetaTolerance,BetaTolerance,GradientTolerance,IterationLimit\n')
    frr.close()
    
    if ".csv" in fname:
        fname = fname.split(".csv")[0]
    ocsvm(fname, parameters, 0)
         
    eng.quit()