rm(list=objects())

setwd("#Your_own_path")

library(tidyverse)
library(lubridate)
library(mgcv)
library(yarrr)
library(magrittr)
library(forecast)

####################  SUBMISSION ############################

#Equation used for the GAM model
equation = Load ~ WeekDays + s(toy,Temp,k=12) +
  te(Load.1,Load.7,k=7)+ s(Load.1,by =Summer_break, bs='cr') +
  s(Load.1,by =Christmas_break,bs='cr')+ s(Load.7, by = Summer_break,bs='cr') +
  s(Temp_s95,Temp_s99) + BH + s(Load.1,by = WeekDays2,bs='cr') +
  Temp_s95_min  + s(Load.7, by = WeekDays2,bs='cr') + 
  s(Load.1,by = Month,bs='cr') + s(Load.7,by = Month,bs='cr')


Data0 <- read_delim("Data/train.csv", delim=",")
Data1<- read_delim("Data/test.csv", delim=",")
Data0$Time = as.numeric(Data0$Date)
Data1$Time = as.numeric(Data1$Date)
Data0$WeekDays2 = forcats::fct_recode(Data0$WeekDays, 'WorkDay'='Thursday',
                                      'Workday'='Tuesday','Workday'='Wednesday')
Data1$WeekDays2 = forcats::fct_recode(Data1$WeekDays, 'WorkDay'='Thursday',
                                      'Workday'='Tuesday','Workday'='Wednesday')

gam_model<-gam(equation, data=Data0)
gam_model.forecast<-predict(gam_model,  newdata= Data1)

submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Load <- gam_model.forecast
#write.table(submit, file="Data/submission_gam.csv", quote=F,sep=",", dec='.',row.names = F)
