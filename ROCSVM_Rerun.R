#importing libraries
suppressMessages(library(tidyverse)) 
suppressMessages(library(rmatio))
suppressMessages(library(dbscan))
suppressMessages(library(MLmetrics))
suppressMessages(library(rio))
#library(raveio))
suppressMessages(library(rlang))
suppressMessages(library(e1071))
suppressMessages(library(pdfCluster))
suppressMessages(library(comprehenr))
suppressMessages(library(mclust))


datasetFolderDir = 'Dataset/'
ocsvm_ = function(filename, kernel, degree, gamma, coef0, tolerance, nu, shrinking, cachesize, epsilon){
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
  
  runOCSVM(filename,X,gt,kernel, degree, gamma, coef0,tolerance,nu,shrinking,cachesize,epsilon)
  
  
}

runOCSVM = function(filename,X,gt,p1, p2, p3, p4, p5, p6, p7, p8, p9){
  labelfile = paste0(filename,"_",p1,"_",p2,"_",p3,"_",p4,"_",p5,"_",p6,"_",p7,"_",p8,"_",p9)
  if (file.exists(paste('Labels/OCSVM_R/',labelfile,".csv",sep=""))){
    return()
  }
  labels = c()
  f1 = c()
  ari = c()

  
  labels_df = data.frame()
  
  if (p3 == "auto"){
    p3 = 1 / dim(X)[2]
  }else{
    p3 = 1 / (dim(X)[2]*mean(var(X)))
  }
  if (p6 == "IF"){
    df_anomaly = read.csv(paste('Stats/SkPercentage.csv',sep="")[1])
    master_files = df_anomaly$Filename
    p6 = df_anomaly[df_anomaly$Filename == filename, ]$IF
  }
  
  
  tryCatch({
    clustering = svm(X,kernel = p1, degree = p2, gamma = p3,coef0 = p4,tolerance = p5,nu = p6,shrinking = p7,cachesize = p8,epsilon = p9)
  }, error = function(err){
    print(err)
    return()
  })
  l = predict(clustering,X)
  list_pred = to_vec(for(j in c(1:length(l))) if(l[j] == TRUE) l[j] = 1 else l[j] = 0)
  
  labels = c(labels,list(list_pred))
  f1score = F1_Score(gt,list_pred)
  f1 = c(f1,f1score)
  labels_df = rbind(labels_df, data.frame(t(sapply(l,c))))
  write.csv(labels_df,paste('Labels/OCSVM_R/',labelfile,".csv",sep=""))
  
  
}
df = read.csv(paste("GD_ReRun/ROCSVM.csv",sep = "")[1])

options(scipen=9999) #############

for (row in 1:nrow(df)) {
  kernel <- df[row, "kernel"]
  degree <- df[row, "degree"]
  gamma <- df[row, "gamma"]
  coef0 <- df[row, "coef0"]
  tolerance <- df[row, "tolerance"]
  nu <- df[row, "nu"]
  shrinking <- df[row, "shrinking"]
  cachesize <- df[row, "cachesize"]
  epsilon <- df[row, "epsilon"]
  ocsvm_(df[row, "Filename"], kernel, degree, gamma, coef0,tolerance,nu,shrinking,cachesize,epsilon)
}

