rm(list=objects())

setwd("C:/Users/alex_/Fac/M1/M1/S2/Modélisation prédictive/Zone de Test")

library(tidyverse)
library(lubridate)
library(mgcv)
library(yarrr)
library(magrittr)
library(forecast)
library(ranger)
library(opera)


####################  SUBMISSION ############################
#bloc CV
block_CV_index = function(Data)
{
  Nblock = 10
  borne_block = seq(1, nrow(Data), length=Nblock+1)%>%floor
  block_list = list()
  l = length(borne_block)
  for (i in c(2:(l-1)))
  {
    block_list[[i-1]] = c(borne_block[i-1]:(borne_block[i]-1))
  }
  block_list[[l-1]] = c(borne_block[l-1]:(borne_block[l]))
  return(block_list)
}

block_CV<-function(equation, block,Data)
{
  g<- gam(as.formula(equation), data=Data[-block,])
  forecast<-predict(g, newdata=Data[block,])
  return(forecast)
} 


equation = Load ~ WeekDays + s(toy,Temp,k=12) +
  te(Load.1,Load.7,k=7)+ s(Load.1,by =Summer_break, bs='cr') +
  s(Load.1,by =Christmas_break,bs='cr')+ s(Load.7, by = Summer_break,bs='cr') +
  s(Temp_s95,Temp_s99) + BH + s(Load.1,by = WeekDays2,bs='cr') + 
  Temp_s95_min  + s(Load.7, by = WeekDays2,bs='cr') + s(Load.1,by = Month,bs='cr') + s(Load.7,by = Month,bs='cr')


Data0 <- read_delim("Data/train.csv", delim=",")
Data1<- read_delim("Data/test.csv", delim=",")
Data0$Time = as.numeric(Data0$Date)
Data1$Time = as.numeric(Data1$Date)
Data0$WeekDays2 = forcats::fct_recode(Data0$WeekDays, 'WorkDay'='Thursday',
                                      'Workday'='Tuesday','Workday'='Wednesday')
Data1$WeekDays2 = forcats::fct_recode(Data1$WeekDays, 'WorkDay'='Thursday',
                                      'Workday'='Tuesday','Workday'='Wednesday')

#Get rid of the Id column
Data1 = Data1[,-20]


block_list1 = blockCV(Data0)

gam_model<-gam(equation, data=Data0)
gam.forecast.sub<-predict(gam_model,  newdata= Data1)

Block_forecast<- predict(gam_model,newdata=Data0)
Block_residuals <- Data0$Load-Block_forecast

Block_residuals.ts <- ts(Block_residuals, frequency=7)
gam.forecast_update <- gam.forecast.sub

Past_Data = tibble(Data0)

for(i in c(1: (nrow(Data1)-1)))
{
  load_result = Data1[i+1,]["Load.1"]
  colnames(load_result)[1]="Load"
  line_merged = cbind(load_result,Data1[i,])
  Past_Data = rbind(Past_Data,line_merged)
  
  gam_model = gam(equation, data=Past_Data)
  Block_residuals <- Past_Data$Load - predict(gam_model,newdata=Past_Data)
  Block_residuals.ts <- ts(Block_residuals, frequency=7)
  
  intermediate <- predict(gam_model, newdata=Data1[i+1,])
  fit.arima.res <- auto.arima(Block_residuals.ts,max.p=3,max.q=4, max.P=2, max.Q=2, trace=T,ic="aic", method="CSS")
  ts_res_forecast <- ts(c(Block_residuals.ts, Data1$Load.1[i+1] - intermediate))
  refit <- Arima(ts_res_forecast, model=fit.arima.res)
  
  prevARIMA.res <- tail(refit$fitted, 1)
  gam.forecast_update[i+1] <- intermediate + prevARIMA.res
  print(i)
}

submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Load <- gam.forecast_update
write.table(submit, file="Data_User/submission_gam_online.csv", quote=F, sep=",", dec='.',row.names = F)

