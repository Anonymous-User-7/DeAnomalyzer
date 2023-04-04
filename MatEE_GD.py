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
import subprocess
import matlab.engine
eng = matlab.engine.start_matlab()

datasetFolderDir = 'Dataset/'

withGT = True

fname = str(sys.argv[1])


def ee(filename, parameters, parameter_iteration):
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
    mod_parameters = deepcopy(parameters)
    ##
    
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

def get_blind_route(X, gt, filename, parameters_this_file, parameter_iteration):
    blind_route = []
    
    for p_i in range(len(parameters_this_file)):
        p = p_i
        
        parameter_route = []
        ari_scores = []
        passing_param = deepcopy(parameters_this_file)

        default_f1, default_ari = runEE(filename, X, gt, passing_param, parameter_iteration)

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
            f1_score, ari_score = runEE(filename, X, gt, passing_param, parameter_iteration)

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

        default_f1, default_ari = runEE(filename, X, gt, passing_param, parameter_iteration)

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
            f1_score, ari_score = runEE(filename, X, gt, passing_param, parameter_iteration)

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
    
    
def runEE(filename, X, gt, params, parameter_iteration):
    global withGT
    labelFile = filename + "_" + str(params[0][1]) + "_" + str(params[1][1]) + "_" + str(params[2][1]) + "_" + str(params[3][1]) + "_" + str(params[4][1]) + "_" + str(params[5][1]) + "_" + str(params[6][1]) + "_" + str(params[7][1]) + "_" + str(params[8][1])
    

    if os.path.exists("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile+".csv") == 0:        
        frr=open("GD_ReRun/MatEE.csv", "a")
        frr.write(filename+","+str(params[0][1])+","+str(params[1][1])+","+str(params[2][1])+","+str(params[3][1])+","+str(params[4][1])+","+str(params[5][1])+","+str(params[6][1])+","+str(params[7][1])+","+str(params[8][1])+'\n')
        frr.close()
        try:
            eng.MatEE_Rerun(nargout=0)
            frr=open("GD_ReRun/MatEE.csv", "w")
            frr.write('Filename,Method,OutlierFraction,NumTrials,BiasCorrection,NumOGKIterations,UnivariateEstimator,ReweightingMethod,NumConcentrationSteps,StartMethod\n')
            frr.close()
            if os.path.exists("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile+".csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0)  
    
    
    f1 = []
    ari = []
    
    
    labels =  pd.read_csv("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile+".csv", header=None).to_numpy()
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

    Method = ["olivehawkins", "fmcd", "ogk"];
    OutlierFraction = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5];
    NumTrials = [2, 5, 10, 25, 50, 100, 250, 500, 750, 1000];
    BiasCorrection = [1, 0];
    NumOGKIterations = [1, 2, 3];
    UnivariateEstimator = ["tauscale", "qn"];
    ReweightingMethod = ["rfch", "rmvn"];
    NumConcentrationSteps = [2, 5, 10, 15, 20];
    StartMethod = ["elemental","classical", "medianball"];    
    
    parameters.append(["Method", "fmcd", Method])
    parameters.append(["OutlierFraction", 0.5, OutlierFraction])
    parameters.append(["NumTrials", 500, NumTrials])
    parameters.append(["BiasCorrection", 1, BiasCorrection])
    parameters.append(["NumOGKIterations", 2, NumOGKIterations])
    parameters.append(["UnivariateEstimator", "tauscale", UnivariateEstimator])
    parameters.append(["ReweightingMethod", "rfch", ReweightingMethod])
    parameters.append(["NumConcentrationSteps", 10, NumConcentrationSteps])
    parameters.append(["StartMethod", "classical", StartMethod])
    
   
    
    frr=open("GD_ReRun/MatEE.csv", "w")
    frr.write('Filename,Method,OutlierFraction,NumTrials,BiasCorrection,NumOGKIterations,UnivariateEstimator,ReweightingMethod,NumConcentrationSteps,StartMethod\n')
    frr.close()
    
    if ".csv" in fname:
        fname = fname.split(".csv")[0]
    ee(fname, parameters, 0)
    eng.quit()
        
        
        
