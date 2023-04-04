import warnings
warnings.filterwarnings('ignore')
import sys
import os
import pandas as pd
import mat73
from scipy.io import loadmat
import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import LocalOutlierFactor

import subprocess
import matlab.engine
eng = matlab.engine.start_matlab()

datasetFolderDir = 'Dataset/'


withGT = True

fname = str(sys.argv[1])


def lof(filename):
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
    
    lof_cont, labels_sk = LOF_ContFactor(X)
    
    f = open("GD_ReRun/LOF.csv", "w")
    f.write(filename+","+str(lof_cont))
    f.close()
    
    ## Default
    if os.path.exists("Labels/LOF_R/"+filename+"_Default.csv") == 0:
        try:
            subprocess.call((["/usr/local/bin/Rscript", "--vanilla", "RLOF_Rerun.r"]))
            if os.path.exists("Labels/LOF_R/"+filename+"_Default.csv") == 0:      
                print("\nFaild to run Rscript from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Rscript from Python.\n")
            exit(0)  
    if os.path.exists("Labels/LOF_Matlab/"+filename+"_Default.csv") == 0:        
        try:
            eng.MatLOF_Rerun(nargout=0)
            if os.path.exists("Labels/LOF_Matlab/"+filename+"_Default.csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0) 
            
    
    labels_r = pd.read_csv("Labels/LOF_R/"+filename+"_Default.csv", header=None).to_numpy()
    labels_r = np.reshape(labels_r, (1, len(labels_r)))
    labels_mat = pd.read_csv("Labels/LOF_Matlab/"+filename+"_Default.csv", header=None).to_numpy()
    
    ari_mvr = adjusted_rand_score(labels_r[0], labels_mat[0])
    ari_rvs = adjusted_rand_score(labels_sk, labels_r[0])
    ari_mvs = adjusted_rand_score(labels_sk, labels_mat[0])
    
    DefaultARI = (ari_mvr + ari_rvs + ari_mvs)/3
    
    print("Default settings: ")
    print("\tMean Mutual-ARI: ", DefaultARI)
    if withGT:
        DefaultF1_r = metrics.f1_score(gt, labels_r[0])
        DefaultF1_mat = metrics.f1_score(gt, labels_mat[0])
        DefaultF1_sk = metrics.f1_score(gt, labels_sk)
        print("\tF1 Score: ")
        print("\t\tR: ", DefaultF1_r)
        print("\t\tMatlab: ", DefaultF1_mat)
        print("\t\tScikit-Learn: ", DefaultF1_sk)
    
    # ## DeAnam
    
    
    if os.path.exists("Labels/LOF_R/"+filename+"_Mod.csv") == 0:
        try:
            subprocess.call((["/usr/local/bin/Rscript", "--vanilla", "RLOF_Rerun.r"]))
            if os.path.exists("Labels/LOF_R/"+filename+"_Mod.csv") == 0:      
                print("\nFaild to run Rscript from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Rscript from Python.\n")
            exit(0)  
    if os.path.exists("Labels/LOF_Matlab/"+filename+"_Mod.csv") == 0:        
        try:
            eng.MatLOF_Rerun(nargout=0)
            if os.path.exists("Labels/LOF_Matlab/"+filename+"_Mod.csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0) 
            
    
    labels_r = pd.read_csv("Labels/LOF_R/"+filename+"_Mod.csv", header=None).to_numpy()
    labels_r = np.reshape(labels_r, (1, len(labels_r)))
    labels_mat = pd.read_csv("Labels/LOF_Matlab/"+filename+"_Mod.csv", header=None).to_numpy()
    
    ari_mvr = adjusted_rand_score(labels_r[0], labels_mat[0])
    ari_rvs = adjusted_rand_score(labels_sk, labels_r[0])
    ari_mvs = adjusted_rand_score(labels_sk, labels_mat[0])
    
    
    UninformedARI = (ari_mvr + ari_rvs + ari_mvs)/3
    
    print("Univariate Search: ")
    print("\tMean Mutual-run ARI: ", UninformedARI)
    if withGT:
        UninformedF1_r = metrics.f1_score(gt, labels_r[0])
        UninformedF1_mat = metrics.f1_score(gt, labels_mat[0])
        UninformedF1_sk = metrics.f1_score(gt, labels_sk)
        print("\tF1 Score: ")
        print("\t\tR: ", UninformedF1_r)
        print("\t\tMatlab: ", UninformedF1_mat)
        print("\t\tScikit-Learn: ", UninformedF1_sk)
    LOF_R_Thr = pd.read_csv("GD_ReRun/LOF_R_Thr.csv", header=None).to_numpy()
    LOF_M_Thr = pd.read_csv("GD_ReRun/LOF_M_Thr.csv", header=None).to_numpy()
    print("\tOutput Parameters:")
    print("\n\tR: Threshold:", LOF_R_Thr[0][0])
    print("\n\tMatlab: Threshold:", LOF_M_Thr[0][0])

def LOF_ContFactor(X):
    labels_lof = LocalOutlierFactor().fit_predict(X)
    labels_lof = [0 if x == 1 else 1 for x in labels_lof]
    num_label = len(labels_lof)
    _, counts_lof = np.unique(labels_lof, return_counts=True)
    lof_per = min(counts_lof)/(num_label)
    
    if lof_per == 1:
        lof_per = 0
    
    return lof_per, labels_lof



if __name__ == '__main__':
    print("\nRunning DeAnomalyzer on ", fname)
    if ".csv" in fname:
        fname = fname.split(".csv")[0]
    lof(fname)
    eng.quit()
    