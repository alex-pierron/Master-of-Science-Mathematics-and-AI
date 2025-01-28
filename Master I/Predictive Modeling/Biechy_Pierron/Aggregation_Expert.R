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

rmse = function(y,ychap,digits = 0)
{
  return ( round(sqrt(mean((y-ychap)^2,na.rm=TRUE)),digits=digits))
}

Data0 = read_delim("Data/train.csv", delim =",")
Data1 = read_delim("Data/test.csv", delim =",")

#Expert Aggregation

#We import all the experts we need
#You can choose wether to use our data or yours by uncommentating or commentating
# the following lines.
gam.forecast.arima = read.csv2("Data_Given/submission_gam_arima.csv",header=T,sep=',')$Load
gam.forecast_alone = read.csv2("Data_Given/submission_gam_alone.csv",header=T,sep=',')$Load
gam.forecast_online = read.csv2("Data_Given/submission_gam_online.csv",header=T,sep=',')$Load
rf.forecast_update = read.csv2("Data_Given/submission_online_forest.csv",header=T,sep=',')$Load
boosting.forecast = read.csv2("Data_Given/submission_boosting.csv",header=T, sep=',')$Load


#gam.forecast.arima = read.csv2("Data_User/submission_gam_arima.csv",header=T,sep=',')$Load
#gam.forecast_alone = read.csv2("Data_User/submission_gam_alone.csv",header=T,sep=',')$Load
#gam.forecast_online = read.csv2("Data_User/submission_gam_online.csv",header=T,sep=',')$Load
#rf.forecast_update = read.csv2("Data_User/submission_online_forest.csv",header=T,sep=',')$Load
#boosting.forecast = read.csv2("Data_User/submission_boosting.csv",header=T, sep=',')$Load


experts <- cbind(gam.forecast_alone,gam.forecast.arima, gam.forecast_online,
                 rf.forecast_update,boosting.forecast)%>%as.matrix
nom_exp <- c("gam", "gam_arima","gam_arima_online", "rf","boosting")
colnames(experts) <- c("gam", "gamarima","gamarimaonline", "rf","boosting")
mode(experts) <- "integer"

y_label = Data1$Load.1[2:275]

or <- oracle(Y=y_label, experts[1:274,])
or$rmse  

colnames(experts) <-  nom_exp
rmse_exp <- apply(experts[1:274,], 2, rmse, y=y_label)
sort(rmse_exp)
cumsum_exp <- apply(y_label-experts[1:274,], 2, cumsum)

#Plotting the exponential cumsum for the residuals for each model
par(mfrow=c(1,1))
K <-ncol(experts)
col <- rev(RColorBrewer::brewer.pal(n = max(min(K,11),4),name = "Spectral"))[1:min(K,11)]
matplot(cumsum_exp, type='l', col=col, lty=1, lwd=2)
par(new=T)
legend("bottomleft", col=col, legend=colnames(experts), lty=1, bty='n',cex=0.4)
title(main="Exponential cumsum for the residual during the test period")
#Using oracle to know if the aggregate is relevant
or <- oracle(Y=y_label, experts[1:274,])
or$rmse

#Doing the actual online prediction by aggregating experts
#3 methods are used
agg_mlpol_lgf<- mixture(Y = y_label, experts = experts[1:274,], model = "MLpol", loss.gradient=FALSE)
summary(agg_mlpol_lgf)

agg_mlpol_lgt <- mixture(Y = y_label, experts = experts[1:274,], model = "MLpol", loss.gradient=TRUE)
summary(agg_mlpol_lgt)

agg_boa <- mixture(Y = y_label, experts = experts[1:274,], model = "BOA", loss.gradient=TRUE)
summary(agg_boa)

plot(agg_mlpol_lgt)

#######bias correction
experts_biais <- cbind(gam.forecast_alone,gam.forecast.arima, gam.forecast_online,
                 rf.forecast_update,boosting.forecast)%>%as.matrix
nom_exp <- c("gam", "gam_arima","gam_arima_online", "rf","boosting")
colnames(experts_biais) <- c("gam", "gamarima","gamarimaonline", "rf","boosting")
mode(experts_biais) <- "integer"
expertsM1000 <- experts_biais-1000
expertsP1000 <- experts_biais+1000
experts_biais <- cbind(experts_biais, expertsM1000, expertsP1000)
colnames(experts_biais) <-c(nom_exp, paste0(nom_exp,  "M"), paste0(nom_exp,  "P"))


cumsum_exp <- apply(y_label-experts_biais[1:274,], 2, cumsum)

par(mfrow=c(1,1))
K <-ncol(experts_biais)
col <- rev(RColorBrewer::brewer.pal(n = max(min(K,11),4),name = "Spectral"))[1:min(K,11)]
matplot(cumsum_exp, type='l', col=col, lty=1, lwd=2)
par(new=T)

legend("topleft", col=col, legend=c(colnames(experts_biais)),lty=1,
       bty ="o",box.lty=2,box.col = "transparent",
       cex=0.27,bg="transparent")


or_biais <- oracle(Y=y_label, experts_biais[1:274,])
or_biais$rmse

#We use the unbiaised experts biaised experts to make the prediction

agg_mlpol_lgf_biais<- mixture(Y = y_label, experts = experts_biais[1:274,], model = "MLpol", loss.gradient=FALSE)
summary(agg_mlpol_lgf_biais)

agg_mlpol_lgt_biais <- mixture(Y = y_label, experts = experts_biais[1:274,], model = "MLpol", loss.gradient=TRUE)
summary(agg_mlpol_lgt_biais)

agg_boa_biais <- mixture(Y = y_label, experts = experts_biais[1:274,], model = "BOA", loss.gradient=TRUE)
summary(agg_boa_biais)

plot(agg_mlpol_lgt_biais)

#We keep the best model from the previous three 

submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Load <- rbind(agg_mlpol_lgt$prediction,gam.forecast.arima[275])
write.table(submit, file="Data_User/submission_expert_aggregate.csv", quote=F, sep=",", dec='.',row.names = F)

