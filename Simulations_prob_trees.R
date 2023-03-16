rm(list =ls())
gc()
set.seed(10)

library(tidyverse)
library(doParallel)
library(bit64)
library(ranger) #random forest package
library(sqldf)
library(data.table)
library(cvAUC)
library(foreach)
library(ROCR)
library(purrr)

 setwd("G:\\ANALYSIS\\ziyi\\try_time")
 load("G:/ANALYSIS/ziyi/try_time/prfaws/AWS_data/data_aws.RData")
 source("G:\\ANALYSIS\\ziyi\\try_time\\prfaws\\Functions\\perf.results.R")
 source("G:\\ANALYSIS\\ziyi\\try_time\\prfaws\\Functions\\sboot.R")
 source("G:\\ANALYSIS\\ziyi\\try_time\\prfaws\\Functions\\sbootprop.R")
 source("G:\\ANALYSIS\\ziyi\\try_time\\prfaws\\Code\\summary_prob.R")


# assumes data is in data frame call "data"
#data$Y is the outcome
#A. Set parameters

Netrue = 1000000 #size of etrue dataset

Nsim = 500 #sim time

#parameters
n_vec = c(100000, 10000, 5000, 1000, 500)
p = 150
full_ntrees_vec = c(100, 50, 25, 10)
full_min_node_vec = c(10, 50, 100, 500, 1000, 5000)
full_mtry_vec = round(sqrt(p))

samp_ntrees_vec = c( 500, 250, 100, 50 ,10)
samp_min_node_vec = c(1, 10, 25, 50, 100)
samp_mtry_vec =c( round(sqrt(p)), round(2*sqrt(p)))

#All types of RFs
all_RFs = c("full_prob","full_class","Ratio1to1_prob","Ratio1to1_class","Ratio1to2_prob",
            "Ratio1to2_class","Ratio1to5_prob","Ratio1to5_class")

#Save running time for each model in each sim
#comp_time = as.data.frame(matrix(NA, nrow=Nsim, ncol=length(all_RFs)))
#as.data.frame(matrix(NA, nrow=10, ncol=40))
#names(comp_time) = all_RFs

#Save performance results
perf.measures = c("AUC", "PPV99","PPV95", "PPV90", "SENS99", "SENS95", "SENS90", 
                  "SPEC99", "SPEC95", "SPEC90" ,'Brier','calibAPR_0%-50%','calibAPR_50%-75%','calibAPR_75%-90%','calibAPR_90%-95%','calibAPR_95%-99%','calibAPR_99%-100%',
                  'calibOR_0%-50%','calibOR_50%-75%','calibOR_75%-90%','calibOR_90%-95%','calibOR_95%-99%','calibOR_99%-100%')
perf.matrix = as.data.frame(matrix(NA, nrow=Nsim, ncol=length(perf.measures)))
names(perf.matrix) = perf.measures 

source("G:\\ANALYSIS\\ziyi\\try_time\\prfaws\\Code\\comp_time_predict.R")
source("G:\\ANALYSIS\\ziyi\\try_time\\prfaws\\Code\\comp_time_ranger.R")
source("G:\\ANALYSIS\\ziyi\\try_time\\prfaws\\Code\\comp_time_perfresult.R")

####################################################################################

#B. Creating places to store the results

#1.Performance results of full samp RFs
res_full = list()

for(iNi in n_vec){
  res_full[[paste("N",iNi,sep="_")]] = list()
  for(iti in full_ntrees_vec){
    res_full[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]] = list()
    
    for( ini in full_min_node_vec){
      res_full[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]] = list()
      
      for( imi in full_mtry_vec){
        res_full[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]][[paste("Mtry",imi,sep="_")]] = perf.matrix
      }
    } 
  }  
}

train_res_full_class = train_res_full_prob = res_full
valid_res_full_class = valid_res_full_prob = res_full
etrue_res_full_class = etrue_res_full_prob = res_full

#2.Performance results of balanced samp RFs

res_samp = list()
for(iNi in n_vec){
  res_samp[[paste("N",iNi,sep="_")]] = list()
  for(iti in samp_ntrees_vec){
    res_samp[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]] = list()
    for( ini in samp_min_node_vec){
      res_samp[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]] = list()
      for( imi in samp_mtry_vec){
        res_samp[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]][[paste("Mtry",imi,sep="_")]] = perf.matrix
      }
    }
  }  
}

train_res_samp_class=train_res_samp_prob=valid_res_samp_class=list()
valid_res_samp_prob=etrue_res_samp_class=etrue_res_samp_prob=list()

#Add ratio
for(iRi in c(1,2,5)){
  train_res_samp_class[[paste("Ratio1to",iRi,"_class",sep="")]] = res_samp
  train_res_samp_prob[[paste("Ratio1to",iRi,"_prob",sep="")]] = res_samp
  
  valid_res_samp_class[[paste("Ratio1to",iRi,"_class",sep="")]] = res_samp
  valid_res_samp_prob[[paste("Ratio1to",iRi,"_prob",sep="")]] = res_samp
  
  etrue_res_samp_class[[paste("Ratio1to",iRi,"_class",sep="")]] = res_samp
  etrue_res_samp_prob[[paste("Ratio1to",iRi,"_prob",sep="")]] = res_samp
  
}
rm(list=c("res_full","res_samp","perf.matrix"))

#3. Saving the predictions themselves
# this is an option - might be easier to keep this, becuase we can always reconstruct any performance measures
# but if it is too big than we cannot store it and need to be very thoughtful about the performance measures we want

#Save for full samp prediction result
pred_valid= pred_train= pred_etrue = list()

for( iNi in n_vec){
  for(iti in full_ntrees_vec){
    pred_train[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]] = list()
    pred_valid[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]] = list()
    pred_etrue[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]] = list()
    for( ini in full_min_node_vec){
      pred_train[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]] = list()
      pred_valid[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]] = list()
      pred_etrue[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]] = list()
      for( imi in full_mtry_vec){
        pred_train[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]][[paste("Mtry",imi,sep="_")]] = matrix(NA,nrow=Nsim, ncol=iNi )
        pred_valid[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]][[paste("Mtry",imi,sep="_")]] = matrix(NA, ncol=iNi, nrow=Nsim)
        pred_etrue[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]][[paste("Mtry",imi,sep="_")]] = matrix(NA, ncol=Netrue, nrow=Nsim)
      }    
    }
  }  
}
train_pred_full_class = train_pred_full_prob = pred_train 
valid_pred_full_class = valid_pred_full_prob = pred_valid
etrue_pred_full_class = etrue_pred_full_prob = pred_etrue

#Save for Balanced samp RFs
pred_samp_train = list()
pred_samp_valid = list()
pred_samp_etrue = list()
for( iNi in n_vec){
  for(iti in samp_ntrees_vec){
    pred_samp_train[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]] = list()
    pred_samp_valid[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]] = list()
    pred_samp_etrue[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]] = list()
    for( ini in samp_min_node_vec){
      pred_samp_train[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]] = list()
      pred_samp_valid[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]] = list()
      pred_samp_etrue[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]] = list()
      for( imi in samp_mtry_vec){
        pred_samp_train[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]][[paste("Mtry",imi,sep="_")]] =  matrix(NA, ncol=iNi, nrow=Nsim)
        pred_samp_valid[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]][[paste("Mtry",imi,sep="_")]] = matrix(NA, ncol=iNi, nrow=Nsim)
        pred_samp_etrue[[paste("N",iNi,sep="_")]][[paste("Ntrees",iti,sep="_")]][[paste("MNode",ini,sep="_")]][[paste("Mtry",imi,sep="_")]] =  matrix(NA, ncol=Netrue, nrow=Nsim)
      }
    }
  }  
}

################################################################
train_pred_samp_class = train_pred_samp_prob = valid_pred_samp_class = list()
valid_pred_samp_prob = etrue_pred_samp_class = etrue_pred_samp_prob = list()

for(iRi in c(1,2,5)){
  train_pred_samp_class[[paste("Ratio1to",iRi,"_class",sep="")]] = pred_samp_train
  train_pred_samp_prob[[paste("Ratio1to",iRi,"_prob",sep="")]] = pred_samp_train
  
  valid_pred_samp_class[[paste("Ratio1to",iRi,"_class",sep="")]] = pred_samp_valid
  valid_pred_samp_prob[[paste("Ratio1to",iRi,"_prob",sep="")]] = pred_samp_valid
  
  etrue_pred_samp_class[[paste("Ratio1to",iRi,"_class",sep="")]] = pred_samp_etrue
  etrue_pred_samp_prob[[paste("Ratio1to",iRi,"_prob",sep="")]] = pred_samp_etrue
  
}


#############################################################################################################

# C. Do the simulations and save results and predictions
set.seed(199)
dat_s <-data_aws
colnames(dat_s)[1]<-'Y'
full_size<-nrow(dat_s)

picked=sample(seq_len(full_size),size=Netrue)
data_etrue<-dat_s[picked,] # put 1 million visits here for the constant truth
Data_home_sim<-dat_s[-picked,] # put remaining visits here to be used for the simulations

unique(data_etrue$Y)

#
st<-Sys.time()
#cl <- makeCluster(detectCores())
#registerDoParallel(cl) #register parallel backend with 'foreach' package

for(ii in 1:Nsim){
  print(ii)
  for (iNi in n_vec){
    # sample data for training and validation from Data_home_sim

    picked2=sample(seq_len(nrow(Data_home_sim)),size=2*iNi)
    out <- Data_home_sim[picked2,]  ##visits with an event
    
    data_train <- out[1:iNi,] 
    data_valid <- out[(iNi+1):nrow(out),] 
   
    training <- data_train %>% data.table()
    feat <- colnames(training[,-1]) #remove columns target variable
    piNi = paste("N",iNi,sep="_") 
    print(piNi)
 
###########################################################################################
## estimate RF with *probability trees with this *full sampling  
    for( ini in full_min_node_vec){
      right_combo_full = (iNi == 500 & ini == 10) | (iNi == 500 & ini == 50) | 
                          (iNi == 1000 & ini == 50) |(ini == 100) | (ini == 500 & iNi >= 1000) |  
                           (ini == 1000 & iNi >= 5000) |(ini == 5000 & iNi >= 10000)
      if(right_combo_full){
       pini = paste("MNode",ini,sep="_")
       print('Fullprop')
       print(pini)
       for( imi in full_mtry_vec){
         pimi = paste("Mtry",imi,sep="_")
         print(pimi)
         #############################################################################################
         # estimate random forest with *probability trees (RF_full_prob) using these parameters on *FULL data_train
         #set.seed(ii)
         list.f <- vector(mode = "list", length = max(full_ntrees_vec)) ## max.num.tree=ntree_oob=500
         for(i in 1:length(list.f)){
           list.f[[i]] <- sboot(training, 'Y')
         }
      
         t1 = Sys.time()
         RF_full_prob <- ranger(dependent.variable.name='Y',  #specify outcome to predict
                         data = training[, c(feat,'Y'), with=FALSE],  #specify predctor data (subset all training data to exclude held-out fold and select only columns with predictors)
                         num.trees = max(full_ntrees_vec),      #specify number of trees
                         mtry=imi,     #specify number of predictors to sample at each split (mtry)
                         importance="none",  #don't calculate variable importance measures
                         write.forest=TRUE,  #save the random forest object
                         probability=TRUE,  #calculate probabilities for terminal nodes (instead of a classificaiton tree that uses majority voting)
                         min.node.size=ini,   #specify minimum node size
                         respect.unordered.factors="partition",   #for categorical variables, consider all grouping of categories (the alternative is that a categorical variable is turning into a number factor and treated like a continuous variable)
                         oob.error=FALSE,   #do not calculate out of bag statistics
                         save.memory=FALSE,   # do not use memory saving options (can't recall why, but they don't work for some part of what we're doing here)
                         inbag=list.f
                         #num.threads = 7
                         ) # does this make the iterations run faster?
 
         #rm(list.f)
         gc()
 
 
         t2 = Sys.time()
         all_results[all_results[,"N"] == iNi& all_results[,"SampType"] == 'full'&all_results[,"MNode"] == ini&all_results[,"Mtry"] == imi,ii+5]=t2-t1
         
         #comp_time[ii,"full_prob"] = t2-t1
         print(t2-t1)
         per_training <- training
         per_valid <- data_valid %>% data.table()
         per_etrue <- data_etrue %>% data.table()
         
         per_training$Y <- as.numeric(levels(per_training$Y))[per_training$Y]
         per_valid$Y <- as.numeric(levels(per_valid$Y))[per_valid$Y]
         per_etrue$Y <- as.numeric(levels(per_etrue$Y))[per_etrue$Y]
         
         # calculate predictions and performance of RF_full_prob on each dataset and save
         for(iti in full_ntrees_vec){
           piti = paste("Ntrees",iti,sep="_")
           print(piti)
           t3 = Sys.time()
           #save prediction
           per_training$pred_cv <- predict(RF_full_prob, data = per_training[,c(feat,'Y'),with = FALSE],num.trees = iti)$predictions[,2]
           per_valid$pred_cv <- predict(RF_full_prob, data = per_valid[,c(feat,'Y'),with = FALSE],num.trees = iti)$predictions[,2]
           per_etrue$pred_cv <- predict(RF_full_prob, data = per_etrue[,c(feat,'Y'),with = FALSE],num.trees = iti)$predictions[,2]

           #train_pred_full_prob[[piNi]][[piti]][[pini]][[pimi]][ii,] = per_training$pred_cv
           #valid_pred_full_prob[[piNi]][[piti]][[pini]][[pimi]][ii,] = per_valid$pred_cv
           #etrue_pred_full_prob[[piNi]][[piti]][[pini]][[pimi]][ii,] = per_etrue$pred_cv
           #x<-as.data.frame(etrue_pred_full_prob[[piNi]][[piti]][[pini]][[pimi]])
           t4 = Sys.time()
           print(t4-t3)
           all_results_predict[all_results_predict[,"N"] == iNi& all_results_predict[,"SampType"] == 'full'&all_results_predict[,"MNode"] == ini&all_results_predict[,"Mtry"] == imi&all_results_predict[,"Ntree"] == iti,ii+6]=t4-t3
           
           t5 = Sys.time()
           #save performance results
           res.train<-perf.results(pred=per_training$pred_cv,
                                   outcome=per_training$Y,
                                   train.pred=per_training$pred_cv,
                                   strata.pctile=c(0.5,0.75,0.90,0.95,0.99))

           res.valid<-perf.results(pred=per_valid$pred_cv,
                                   outcome=per_valid$Y,
                                   train.pred=per_training$pred_cv,
                                   strata.pctile=c(0.5,0.75,0.90,0.95,0.99))

           res.etrue<-perf.results(pred=per_etrue$pred_cv,
                                   outcome=per_etrue$Y,
                                   train.pred=per_training$pred_cv,
                                   strata.pctile=c(0.5,0.75,0.90,0.95,0.99))
           t6 = Sys.time()
           print(t6-t5)
           all_results_perf.result[all_results_predict[,"N"] == iNi& all_results_perf.result[,"SampType"] == 'full'&all_results_perf.result[,"MNode"] == ini&all_results_perf.result[,"Mtry"] == imi&all_results_perf.result[,"Ntree"] == iti,ii+6]=t6-t5
           
       

           perf.train <- as.data.frame(t(c(res.train$AUC,
                                           res.train$Performance$PPV[3:5],
                                           res.train$Performance$Sens[3:5],
                                           res.train$Performance$Spec[3:5],
                                           res.train$`Brier Score`,
                                           res.train$Calibration$AvgPredRisk,
                                           res.train$Calibration$ObsRisk)))
           perf.valid <- as.data.frame(t(c(res.valid$AUC,
                                           res.valid$Performance$PPV[3:5],
                                           res.valid$Performance$Sens[3:5],
                                           res.valid$Performance$Spec[3:5],
                                           res.valid$`Brier Score`,
                                           res.valid$Calibration$AvgPredRisk,
                                           res.valid$Calibration$ObsRisk)))
           print(perf.valid)
           perf.etrue <- as.data.frame(t(c(res.etrue$AUC,
                                           res.etrue$Performance$PPV[3:5],
                                           res.etrue$Performance$Sens[3:5],
                                           res.etrue$Performance$Spec[3:5],
                                           res.etrue$`Brier Score`,
                                           res.etrue$Calibration$AvgPredRisk,
                                           res.etrue$Calibration$ObsRisk)))

           train_res_full_prob[[piNi]][[piti]][[pini]][[pimi]][ii,] = perf.train
           valid_res_full_prob[[piNi]][[piti]][[pini]][[pimi]][ii,] = perf.valid
           etrue_res_full_prob[[piNi]][[piti]][[pini]][[pimi]][ii,] = perf.etrue
          } # end iti in full_ntrees_vec
         } # end imi in full_mtry_vec
      } # close if(right_combo_full)
     } #end ini in full_min_node_vec
     # 
    
###########################################################################################
 ## estimate RF with *probabilty trees with this *sampling ratio
    for( ini in samp_min_node_vec){
       pini = paste("MNode",ini,sep="_")
       print('sampling prob')
       print(pini)
       ## not right
       for( imi in samp_mtry_vec){
         mimi = paste("Mtry",imi,sep="_")
         print(mimi)
         ## loop through sample ratio to use sampled RF
         for(iRi in c(1,2,5)){
          right_combo_samp = (ini == 1) | (ini == 10 & iRi == 1 & iNi >= 1000) | 
                              (ini == 25 & iRi == 1 & iNi >= 10000) |     
                              (ini == 10 & iRi == 2 & iNi >= 1000) |       
                              (ini == 25 & iRi == 2 & iNi >= 10000) |       
                              (ini == 50 & iRi == 2 & iNi >= 10000) |       
                              (ini == 10 & iRi == 5 ) |       
                              (ini == 25 & iRi == 5 & iNi >= 1000) |       
                              (ini == 50 & iRi == 5 & iNi >= 10000) |       
                              (ini == 100 & iRi == 5 & iNi >= 10000)        
          if(right_combo_samp){
           piRi = paste("Ratio1to",iRi,"_prob",sep="")
           print(piRi)
           list.f2 <- vector(mode = "list", length = max(samp_ntrees_vec)) ## max.num.tree=ntree_oob=500
           
           for(i in 1:length(list.f2)){
             list.f2[[i]] <- sbootprop(training, "Y",iRi)
           }
           t1 = Sys.time()
           RF_samp_prob <- ranger(dependent.variable.name="Y",  #specify outcome to predict
                                  data = training[, c(feat,"Y"), with=FALSE],  #specify predctor data (subset all training data to exclude held-out fold and select only columns with predictors)
                                  num.trees = max(samp_ntrees_vec),      #specify number of trees
                                  mtry=imi,     #specify number of predictors to sample at each split (mtry)
                                  importance="none",  #don't calculate variable importance measures
                                  write.forest=TRUE,  #save the random forest object
                                  probability=TRUE,  #calculate probabilities for terminal nodes (instead of a classificaiton tree that uses majority voting)
                                  min.node.size=ini,   #specify minimum node size
                                  respect.unordered.factors="partition",   #for categorical variables, consider all grouping of categories (the alternative is that a categorical variable is turning into a number factor and treated like a continuous variable)
                                  oob.error=FALSE,   #do not calculate out of bag statistics
                                  save.memory=FALSE,   # do not use memory saving options (can't recall why, but they don't work for some part of what we're doing here)
                                  inbag=list.f2
                                  #num.threads = 7
                                  )
  
           rm(list.f2)
           gc()
  
           t2 = Sys.time()

           print(t2-t1)
           all_results[all_results[,"N"] == iNi& all_results[,"SampType"] == piRi &all_results[,"MNode"] == ini &all_results[,"Mtry"] == imi, ii+5]=t2-t1
         
           #comp_time[ii,piRi] = t2-t1
           # #prep for results
           per_training <- training
           per_valid <- data_valid %>% data.table()
           per_etrue <- data_etrue %>% data.table()

           per_training$Y <- as.numeric(levels(per_training$Y))[per_training$Y]
           #print(per_training$pred_cv[1:10])
           per_valid$Y <- as.numeric(levels(per_valid$Y))[per_valid$Y]
           per_etrue$Y <- as.numeric(levels(per_etrue$Y))[per_etrue$Y]

           for(iti in samp_ntrees_vec){
             piti = paste("Ntrees",iti,sep="_")
             print(piti)
             
             #save prediction
             t3 = Sys.time()
             per_training$pred_cv2 <- predict(RF_samp_prob, data = per_training[,c(feat,'Y'),with = FALSE],num.trees = iti)$predictions[,2]
             per_valid$pred_cv2 <- predict(RF_samp_prob, data = per_valid[,c(feat,'Y'),with = FALSE],num.trees = iti)$predictions[,2]
             per_etrue$pred_cv2 <- predict(RF_samp_prob, data = per_etrue[,c(feat,'Y'),with = FALSE],num.trees = iti)$predictions[,2]
             t4 = Sys.time()
             print(t4-t3)
             all_results_predict[all_results_predict[,"N"] == iNi& all_results_predict[,"SampType"] == piRi&all_results_predict[,"MNode"] == ini&all_results_predict[,"Mtry"] == imi&all_results_predict[,"Ntree"] == iti,ii+6]=t4-t3
             
             #x<-as.data.frame(etrue_pred_samp_prob[[piRi]][[piNi]][[piti]][[pini]][[pimi]])

             ## save predictions from the RF created with sampling on ALL of the data in each of the data sets
             #train_pred_samp_prob[[piRi]][[piNi]][[piti]][[pini]][[mimi]][ii,] = per_training$pred_cv2
             #valid_pred_samp_prob[[piRi]][[piNi]][[piti]][[pini]][[mimi]][ii,] = per_valid$pred_cv2
             #etrue_pred_samp_prob[[piRi]][[piNi]][[piti]][[pini]][[mimi]][ii,] = per_etrue$pred_cv2

             t5 = Sys.time()
             #save performance results
             res.train2<-perf.results(pred=per_training$pred_cv2,
                                     outcome=per_training$Y,
                                     train.pred=per_training$pred_cv2,
                                     strata.pctile=c(0.5,0.75,0.90,0.95,0.99))

             res.valid2<-perf.results(pred=per_valid$pred_cv2,
                                     outcome=per_valid$Y,
                                     train.pred=per_training$pred_cv2,
                                     strata.pctile=c(0.5,0.75,0.90,0.95,0.99))

             res.etrue2<-perf.results(pred=per_etrue$pred_cv2,
                                     outcome=per_etrue$Y,
                                     train.pred=per_training$pred_cv2,
                                     strata.pctile=c(0.5,0.75,0.90,0.95,0.99))
             t6 = Sys.time()
             print(t6-t5)
             all_results_perf.result[all_results_predict[,"N"] == iNi& all_results_perf.result[,"SampType"] == piRi&all_results_perf.result[,"MNode"] == ini&all_results_perf.result[,"Mtry"] == imi&all_results_perf.result[,"Ntree"] == iti,ii+6]=t6-t5
             
             perf.train2 <- as.data.frame(t(c(res.train2$AUC,
                                             res.train2$Performance$PPV[3:5],
                                             res.train2$Performance$Sens[3:5],
                                             res.train2$Performance$Spec[3:5],
                                             res.train2$`Brier Score`,
                                             res.train2$Calibration$AvgPredRisk,
                                             res.train2$Calibration$ObsRisk)))
             perf.valid2 <- as.data.frame(t(c(res.valid2$AUC,
                                             res.valid2$Performance$PPV[3:5],
                                             res.valid2$Performance$Sens[3:5],
                                             res.valid2$Performance$Spec[3:5],
                                             res.valid2$`Brier Score`,
                                             res.valid2$Calibration$AvgPredRisk,
                                             res.valid2$Calibration$ObsRisk)))
             print(perf.valid2)
             perf.etrue2 <- as.data.frame(t(c(res.etrue2$AUC,
                                             res.etrue2$Performance$PPV[3:5],
                                             res.etrue2$Performance$Sens[3:5],
                                             res.etrue2$Performance$Spec[3:5],
                                             res.etrue2$`Brier Score`,
                                             res.etrue2$Calibration$AvgPredRisk,
                                             res.etrue2$Calibration$ObsRisk)))

             ## save results from the RF created with sampling
             train_res_samp_prob[[piRi]][[piNi]][[piti]][[pini]][[mimi]][ii,] = perf.train2
             valid_res_samp_prob[[piRi]][[piNi]][[piti]][[pini]][[mimi]][ii,] = perf.valid2
             etrue_res_samp_prob[[piRi]][[piNi]][[piti]][[pini]][[mimi]][ii,] = perf.etrue2
            } # end iti in full_ntrees_vec

           } # end if(right_combo_samp){
        }  # iRi in c(1,2,5)
      } # end  imi in samp_mtry_vec
    } # end ini in samp_min_node_vec

   } # end iNi in n_vec
  if( (ii %% 1) == 0){
    save(train_pred_full_prob,file ='train_pred_full_prob.rdata')
    save(valid_pred_full_prob,file ='valid_pred_full_prob.rdata')
    save(etrue_pred_full_prob,file ='etrue_pred_full_prob.rdata')
    save(train_res_full_prob,file ='train_res_full_prob.rdata')
    save(valid_res_full_prob,file ='valid_res_full_prob.rdata')
    save(etrue_res_full_prob,file ='etrue_res_full_prob.rdata')
    save(train_pred_samp_prob,file ='train_pred_samp_prob.rdata')
    save(valid_pred_samp_prob,file ='valid_pred_samp_prob.rdata')
    save(etrue_pred_samp_prob,file ='etrue_pred_samp_prob.rdata')
    save(train_res_samp_prob,file ='train_res_samp_prob.rdata')
    save(valid_res_samp_prob,file ='valid_res_samp_prob.rdata')
    save(etrue_res_samp_prob,file ='etrue_res_samp_prob.rdata')
    }
 } # ii in 1:Nsim
#stopCluster(cl)  
en<-Sys.time()
en-st


write.csv(all_results,file ='comp_time_ranger.csv')
write.csv(all_results_predict,file ='comp_time_predict.csv')
write.csv(all_results_perf.result,file ='comp_time_perfresult.csv')

train_sum<-summary_results(train_res_full_prob,train_res_samp_prob)
write.csv(train_sum,file='train_summary.csv')

valid_sum<-summary_results(valid_res_full_prob,valid_res_samp_prob)
write.csv(valid_sum,file='valid_summary.csv')

etrue_sum<-summary_results(etrue_res_full_prob,etrue_res_samp_prob)
write.csv(etrue_sum,file='etrue_summary.csv')
