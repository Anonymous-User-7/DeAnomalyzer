suppressMessages(library(tidyverse))
suppressMessages(library(rmatio))
suppressMessages(library(dbscan))
suppressMessages(library(rio))
suppressMessages(library(raveio))
suppressMessages(library(MLmetrics))

localoutlierfactor = function(filename,anomaly_perc){
  folderpath = 'Dataset/'
  if (file.exists(paste(folderpath,filename,".mat",sep = ""))){
    df = read_mat(paste(folderpath,filename,".mat",sep = "")[1])
    target = df$y
    X = df$X
  }
  else if (file.exists(paste(folderpath,filename,".csv",sep = ""))){
    df = read.csv(paste(folderpath,filename,".csv",sep = "")[1])
    target = df$target
    X = subset(df, select=-c(target))
  }
  pred = lof(X, minPts=20)
  
  #binary search
  start = min(pred)
  end = max(pred)
  iteration = 1
  while (TRUE) {
    iteration = iteration+1
    if(iteration>1000){
      return_list = list("threshold" = 0, "labels" = pred)
      return(return_list)
      break
    }
    middle = (start+end)/2
    pred_new = pred
    for (j in c(1:length(pred))){
      if(pred[j]> middle){
        pred_new[j] = 1
      }
      else{
        pred_new[j] = 0
      }
    }
    anomaly = (sum(pred_new == 1)/length(pred_new))
    if (abs(anomaly - anomaly_perc) < 0.000001){
      return_list = list("threshold" = middle, "labels" = pred)
      return(return_list)
      break
    }
    else if (anomaly > anomaly_perc){
      start = middle
    }
    else{
      end = middle
    }
  }
}

df_anomaly = read.csv("GD_ReRun/LOF.csv", header = FALSE)
filename = df_anomaly$V1
anomaly_perc = df_anomaly$V2

return_list = localoutlierfactor(filename,anomaly_perc)
thr = return_list$threshold
pred = return_list$labels
#default
pred_def = pred
for (j in c(1:length(pred_def))){
  if(pred_def[j]> 1){
    pred_def[j] = 1
  }
  else{
    pred_def[j] = 0
  }
}

write.table(pred_def, paste("Labels/LOF_R/",filename,"_Default.csv",sep = ""),row.names = FALSE, col.names = FALSE,  sep=",")

#modified
pred_mod = pred
for (j in c(1:length(pred_mod))){
  if(pred_mod[j]> thr){
    pred_mod[j] = 1
  }
  else{
    pred_mod[j] = 0
  }
}

if (thr == 0){
  write.table(pred_def, paste("Labels/LOF_R/",filename,"_Mod.csv",sep = ""),row.names = FALSE, col.names = FALSE,  sep=",")
}else{ 
  write.table(pred_mod, paste("Labels/LOF_R/",filename,"_Mod.csv",sep = ""),row.names = FALSE, col.names = FALSE,  sep=",")
}

write(thr,file="GD_ReRun/LOF_R_Thr.csv",append=FALSE)

