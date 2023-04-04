#importing libraries
library(tidyverse)
library(rmatio)
library(dbscan)
library(MLmetrics)
library(rio)
# library(raveio)
library(rlang)
library(e1071)
library(pdfCluster)
library(comprehenr)
library(mclust)
library(isotree)


datasetFolderDir = 'Dataset/'
isolationforest = function(filename, ntrees, standardize_data, sample_size, ncols_per_tree){
  print(filename)
  folderpath = datasetFolderDir
  if (file.exists(paste(folderpath,filename,".mat",sep = ""))){
    df = read.mat(paste(folderpath,filename,".mat",sep = "")[1])
    gt = df$y
    X = df$X
    if (any(is.na(X))){
      print("Didn't run - NaN")
      return()
    }
  }
  else if (file.exists(paste(folderpath,filename,".csv",sep = ""))){

    df = read.csv(paste(folderpath,filename,".csv",sep = "")[1])
    gt = df$target
    X = subset(df, select=-c(target))
    if (any(is.na(X))){
      print("Didn't run - NaN")
      return()
    }
  }
 
  runif(filename,X,gt,ntrees, standardize_data, sample_size, ncols_per_tree)
      
  
}

runif = function(filename,X,gt,p1, p2, p3, p4){
  labelfile = paste0(filename,"_",p1,"_",p2,"_",p3,"_",p4)
  if (file.exists(paste('Labels/IF_R/',labelfile,".csv",sep=""))){
    return()
  }
  labels = c()
  f1 = c()
  ari = c()
  if (p3 == "auto"){
      p3 = min(nrow(X),10000L)
  }else if(is.null(p3)){
      p3 = p3
  }else{
    p3 = as.double(p3)
  }
  if (p4 == "def"){
      p4 = ncol(X)
  }else{
      p4 = as.double(p4)
  }
  
  
  labels_df = data.frame()
  for (i in 1:c(10)){
    # tryCatch({
      clustering = isolation.forest(X,ntrees = p1,standardize_data = p2, sample_size = p3, ncols_per_tree = p4, seed = sample(c(1:100),1))
    # }, error = function(err){
    #   print(err)
    #   return()
    # })
    l = predict(clustering,X)
    list_pred = to_vec(for(j in c(1:length(l))) if(l[j] > 0.5) l[j] = 1 else l[j] = 0)
    #    for (j in c(1:length(l))){
    #      if(l[j] == TRUE){
    #        l[j] = 1
    #      }
    #      else{
    #        l[j]  = 0
    #      }
    #    }
    labels = c(labels,list(list_pred))
    
    labels_df = rbind(labels_df, data.frame(t(sapply(list_pred,c))))
  }
  write.csv(labels_df,paste('Labels/IF_R/',labelfile,".csv",sep=""))
  
  
}
df = read.csv(paste("GD_ReRun/RIF.csv",sep = "")[1])

for (row in 1:nrow(df)) {
  print(row)
  ntrees <- df[row, "ntrees"]
  standardize_data  <- df[row, "standardize_data"]
  sample_size <- df[row, "sample_size"]
  ncols_per_tree<- df[row, "ncols_per_tree"]
  isolationforest(df[row, "Filename"], ntrees, standardize_data, sample_size, ncols_per_tree)
}

