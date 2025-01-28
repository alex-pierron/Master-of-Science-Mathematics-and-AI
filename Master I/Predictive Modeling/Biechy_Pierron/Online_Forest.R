rm(list=objects())

setwd("#Your_own_path")

library(tidyverse)
library(lubridate)
library(mgcv)
library(yarrr)
library(magrittr)
library(forecast)
library(ranger)
library(opera)
library(xts)

rmse = function(y,ychap,digits = 0)
{
  return ( round(sqrt(mean((y-ychap)^2,na.rm=TRUE)),digits=digits))
}

####################  SUBMISSION ############################
set.seed(2023)

Data0 <- read_delim("Data/train.csv", delim=",")
Data1<- read_delim("Data/test.csv", delim=",")
range(Data1$Date)
range(Data0$Date)
Data0$Time = as.numeric(Data0$Date)
Data1$Time = as.numeric(Data1$Date)
Data0$DLS = as.factor(Data0$DLS)
Data1$DLS = as.factor(Data1$DLS)
Data0$WeekDays2 = forcats::fct_recode(Data0$WeekDays, 'WorkDay'='Thursday',
                                      'Workday'='Tuesday','Workday'='Wednesday')
Data1$WeekDays2 = forcats::fct_recode(Data1$WeekDays, 'WorkDay'='Thursday',
                                      'Workday'='Tuesday','Workday'='Wednesday')


equation_forest <- Load~  Time + toy + Temp + Load.1 + Load.7 + Temp_s99 + 
  WeekDays + BH + Temp_s95_max + 
  Temp_s99_max + Summer_break  + Christmas_break  +
  Temp_s95_min +Temp_s99_min + DLS + GovernmentResponseIndex

num.trees <- 500
mtry = 13
Data1 = Data1[,-20]

#bloc CV
Nblock = 10
borne_block = seq(1, nrow(Data0), length=Nblock+1)%>%floor
block_list = list()
l = length(borne_block)
for (i in c(2:(l-1)))
{
  block_list[[i-1]] = c(borne_block[i-1]:(borne_block[i]-1))
}
block_list[[l-1]] = c(borne_block[l-1]:(borne_block[l]))


rf <- ranger(equation_forest, data=Data0, importance =  'permutation', mtry=mtry, num.trees=num.trees)

rf.forecast <-  predict(rf, data=Data1)$predictions

rf.forecast_update <- rf.forecast

Past_Data = data_frame(Data0)

for(i in c(1: (nrow(Data1)-1)))
{
  load_result = Data1[i+1,]["Load.1"]
  colnames(load_result)[1]="Load"
  line_merged = cbind(load_result,Data1[i,])
  Past_Data = rbind(Past_Data,line_merged)
  colnames(line_merged)
  colnames(Data0)
  rf<- ranger::ranger(equation_forest, data=Past_Data, num.trees =num.trees, mtry=mtry)
  rf.forecast_update[i+1] <- predict(rf, data=Data1[1:i+1,], predict.all = F)$predictions%>%tail(1)
  print(i)
  }

submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Load <- rf.forecast_update
write.table(submit, file="Data_User/submission_online_forest.csv", quote=F, sep=",", dec='.',row.names = F)

