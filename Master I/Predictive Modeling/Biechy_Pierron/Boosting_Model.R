
rm(list=objects())
set.seed(2023)

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
library(gbm)
library(caret)

#### Submission ####

Data0 <- read_delim("Data/train.csv", delim=",")
Data1<- read_delim("Data/test.csv", delim=",")
range(Data1$Date)
range(Data0$Date)
Data0$Date = as.numeric(Data0$Date)
Data0$Time = as.numeric(Data0$Date)
Data1$Time = as.numeric(Data1$Date)
Data1$Date = as.numeric(Data1$Date)
Data0$DLS = as.factor(Data0$DLS)
Data1$DLS = as.factor(Data1$DLS)
Data0$WeekDays2 = as.factor(forcats::fct_recode(Data0$WeekDays, 'WorkDay'='Thursday',
                                                'Workday'='Tuesday','Workday'='Wednesday'))
Data1$WeekDays2 = as.factor(forcats::fct_recode(Data1$WeekDays, 'WorkDay'='Thursday',
                                                'Workday'='Tuesday','Workday'='Wednesday'))
Data0$WeekDays= as.factor(Data0$WeekDays)
Data1$WeekDays= as.factor(Data1$WeekDays)

#Deleting the "Id" column to make sure Data1 and Data0 have the same variables
Data1 = Data1[,-20]

train_x = Data0[, -2] # feature and target array
train_y =Data0[, 2]

equation_boosting = Load~  Time + toy + Temp + Load.1 + Load.7 + Temp_s99 + WeekDays + 
  BH + Temp_s95_max + 
  Temp_s99_max + Summer_break  + Christmas_break  + (Temp**2) +Load.1**2 + toy**2+
  Temp_s95_min +Temp_s99_min + DLS + GovernmentResponseIndex

model_gbm = gbm(formula = equation_boosting,data = Data0,distribution="gaussian",
                cv.folds = 10,shrinkage=0.01,
                n.trees = 1500)

pred_y = predict.gbm(model_gbm, Data1)

submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Load <- pred_y
write.table(submit, file="Data_User/submission_boosting.csv", quote=F, sep=",", dec='.',row.names = F)
