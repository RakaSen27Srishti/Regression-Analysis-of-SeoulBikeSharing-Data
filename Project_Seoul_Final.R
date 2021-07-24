rm(list=ls())
set.seed(1234)
#Installing necessary packages
library(Matrix) # For matrix computations
library(olsrr) # For Variable selection algorithms
library(car) # For VIF

#Reading Data
library(readxl)
SeoulBikeRental <- read_excel(file.choose())
View(SeoulBikeRental) #8760 observations of 14 variables
df=SeoulBikeRental[-c(1,12:14)] #We omit the columns having non-numeric data
View(df)#8760 observations of 10 variables(numeric)
data=data.frame(df)
n=length(df$`Rented Bike Count`) #We have 8760 observations.


head(data)
tail(data)
str(data)
sum(is.null(data))
dim(data)
library(dplyr)
glimpse(data)
summary(data)
sum(duplicated(data))
data_num=data 
head(data_num)
col=colnames(data_num)
y=data[,1]
head(y)

##scatterplots
par(mfrow=c(3,3))
for(i in 1:9)
{
  plot(data_num[,(i+1)],y,xlab=col[i+1],ylab=col[1])
}


#correlation
data_cor=cor(data_num)
data_cor
install.packages("corrplot")
library(corrplot)
corrplot(data_cor,  method = "color", addCoef.col = "black") ## High positive correlation between Dew Point Temperature and Temperature


##outlier
par(mfrow=c(4,3))
for(i in 1:10)
{
  boxplot(data_num[i],data=data_num,col="blue",ylab=col[i])
}
##Huge number of outliers are present in the response variable, Solar Radiation,Rainfall,Snowfall and Windspeed variables

##normality check
par(mfrow=c(4,3))
for(i in 1:10)
{
  hist(data_num[,i],freq=FALSE,main=col[i], xlab=col[i])
  dx=density(data_num[,i])
  lines(dx,lwd=2,col="red")
}
##none of the variables seem to have a Normal distribution


########################################################################



#train-test split
#Getting Row numbers for the training data

library(caret)
library(lattice)
library(ggplot2)

set.seed(1234)
training.samples=createDataPartition(df$`Rented Bike Count`, p=0.8, list=FALSE)
#Creating the training data set
train=df[training.samples,] #7009 observations of 10 variables
#Creating the test data set
test=df[-training.samples,]#1751 observations of 10 variables

#Scaling train data, defining predictors and response
y=train$`Rented Bike Count`
data_pred = train[,-c(1)]
n1=length(train$`Rented Bike Count`)
x_scaled_train = data.frame(sqrt(1/(n1-1))*scale.default(data_pred, center=TRUE,scale=TRUE))
xcs=as.matrix(x_scaled_train)
y_scaled_train=(y-mean(y))/(sqrt(n1-1)*(sd(y)))

#Scaling test data,defining predictors and response
y_test=test$`Rented Bike Count`
data_pred2 = test[,-c(1)]
n2=length(test$`Rented Bike Count`)
x_scaled_test = data.frame(sqrt(1/(n2-1))*scale.default(data_pred2, center=TRUE,scale=TRUE))
xcs2=as.matrix(x_scaled_test)
y_scaled_test=(y_test-mean(y_test))/(sqrt(n2-1)*(sd(y_test)))



#fitting MLRM with CS data
scaled_fit = lm(y_scaled_train~.-1,data=x_scaled_train)
summary(scaled_fit) ###rse=0.008681, adj r2=0.4718

#checking model adequacy on test data
prediction=predict(scaled_fit,x_scaled_test)
library(Metrics)
response=y_scaled_test
rmse(y_scaled_test,prediction) ###rmse= 0.01749787

##Plotting the predicted and observed values
library(lattice)
library(latex2exp)

xyplot(y_scaled_test[1:100] + prediction[1:100] ~ 1:100,
       data = data,
       type = c("l"),
       ylab = "Response & Predicted Values",
       xlab = "Index",
       main = list(
         "Plot of Predicted Values from full model using OLS",
         cex = 1.2
       ),
       auto.key = list(
         space = "top",
         columns = 2,
         text = c(
           "Response Values",
           "Predicted Values"
         )
       ))


#Checking for influencial points, leverage point and outliers:
ols_plot_cooksd_chart(scaled_fit)
ols_plot_resid_stand(scaled_fit)
ols_plot_resid_stud(scaled_fit)
h_ii=lm.influence(scaled_fit)$hat
plot(seq(1,n1,1),h_ii,pch=,xlab="Observation",ylab="Leverages",
     main="Leverage Plot")

# correlation matrix
corr_mat=t(xcs)%*%(xcs)
corr_mat
# Determinant of X'X
det(t(xcs)%*%xcs) #equaling 0.00202548 (multicollinearity is suspected)

# Computing VIF
round(vif(scaled_fit),2)
##Some of the values are >10, so multicollinearity is present.

#Ridge Regression
# Loading the library
library(glmnet)
# Getting the independent variable
x_var <- data.matrix(x_scaled_train)
# Getting the dependent variable
y_var =y_scaled_train
# Setting the range of lambda values
lambda_seq <- 10^seq(2, -2, by = -.1)
fit <- glmnet(x_var, y_var, alpha = 0,standardize = FALSE,standardize.response = FALSE, lambda  = lambda_seq)
summary(fit)
# Using cross validation glmnet
ridge_cv <- cv.glmnet(x_var, y_var, alpha = 0, lambda = lambda_seq)
# Best lambda value
best_lambda <- ridge_cv$lambda.min
best_lambda #best lambda=0.01
# Rebuilding the model with optimal lambda value
best_ridge= glmnet(x_var, y_var, alpha = 0, lambda =0.01)
pred_ridge <- predict(best_ridge, s = best_lambda, newx =data.matrix(x_scaled_test)) 
rmse(y_scaled_test,pred_ridge) ###rmse=0.01865337

##Plotting the predicted and observed values
library(lattice)
library(latex2exp)

xyplot(y_scaled_test[1:100] + pred_ridge[1:100] ~ 1:100,
       data = data,
       type = c("l"),
       ylab = "Response & Predicted Values",
       xlab = "Index",
       main = list(
         "Plot of Predicted and observed Values from ridge regression",
         cex = 1.2
       ),
       auto.key = list(
         space = "top",
         columns = 2,
         text = c(
           "Response Values",
           "Predicted Values"
         )
       ))
# PC Regression
library(pls)
pcr_model <- pcr(y_scaled_train~., data=train[,-c(1)],scale =TRUE, validation = "CV")
summary(pcr_model)
validationplot(pcr_model)
pcr_pred <- predict(pcr_model,test,ncomp = 5)
rmse(y_scaled_test,pcr_pred) #rmse=0.0196237
##Plotting the predicted and observed values
library(lattice)
library(latex2exp)

xyplot(y_scaled_test[1:100] + pcr_pred[1:100] ~ 1:100,
       data = data,
       type = c("l"),
       ylab = "Response & Predicted Values",
       xlab = "Index",
       main = list(
         "Plot of Predicted and observed Values from Principal Component Regression",
         cex = 1.2
       ),
       auto.key = list(
         space = "top",
         columns = 2,
         text = c(
           "Response Values",
           "Predicted Values"
         )
       ))


#Variable reduction
#Forward, Backward and Stepwise Regression


#forward with AIC as the selection criteria
f1=ols_step_forward_aic(scaled_fit,details = TRUE)
f1
prediction3=predict(f1$model,x_scaled_test)
rmse(y_scaled_test,prediction3)##rmse=0.01750209
#Wind Speed and Dew Point temperature is dropped.

#Backward selection with AIC as the selection criteria
f2=ols_step_backward_aic(scaled_fit,details = TRUE)
f2
summary(f2$model)
prediction4=predict(f2$model,x_scaled_test)
rmse(y_scaled_test,prediction4) #0.01749787

#Wind Speed and Dew Point temperature is dropped.

#Step-wise selection with AIC as the selection criteria
f3=ols_step_both_aic(scaled_fit,details = TRUE)
f3
pred5=predict(f3$model,x_scaled_test)
rmse(y_scaled_test,pred5) #0.1749787

#Final variables: Temp, Humidity, Rainfall, Solar radiation, Visibility,snowfall
#So we drop Wind speed, dew point.

#Thus we drop the variables "Wind Speed" and "Dew point temperature".
red_train_x=subset(x_scaled_train,select=-c(4,6))
red_test_x=subset(x_scaled_test,select=-c(4,6))
# fitting MLRM with CS data
reduced_model = lm(y_scaled_train~.-1,data=red_train_x)
summary(reduced_model) #rse=0.008681
#checking model performance on test data
red.pred=predict(reduced_model,red_test_x)
rmse(y_scaled_test,red.pred) #RMSE=0.01750209
##Plotting the predicted and observed values
library(lattice)
library(latex2exp)

xyplot(y_scaled_test[1:100] + red.pred[1:100] ~ 1:100,
       data = data,
       type = c("l"),
       ylab = "Response & Predicted Values",
       xlab = "Index",
       main = list(
         "Plot of Predicted and observed Values from reduced model",
         cex = 1.2
       ),
       auto.key = list(
         space = "top",
         columns = 2,
         text = c(
           "Response Values",
           "Predicted Values"
         )
       ))

############ Outlier detection ####################

lev1=ols_plot_resid_lev(reduced_model) # Computing leverage measure for all observations

# leverage points
which(lev1$data$color=="leverage") 
length(which(lev1$data$color=="leverage")) # number of leverage points 227 

# outliers
which(lev1$data$color=="outlier") 
length(which(lev1$data$color=="outlier")) #number of outliers 386

# outlier and leverage
which(lev1$data$color=="outlier & leverage")
length(which(lev1$data$color=="outlier & leverage")) #number of points which are both outlier and leverage =4

#Cook's distance
lev2=ols_plot_cooksd_bar(reduced_model) # Plotting  Cook's distance 
which(lev2$data$color=="outlier")
length(which(lev2$data$color=="outlier")) # 236

lev3=ols_plot_dffits(reduced_model) #Plotting  DFFFITS statistics
which(lev3$data$color=="outlier")
length(which(lev3$data$color=="outlier"))

ols_plot_dfbetas(reduced_model)# Plotting DFBETAS statistics

design_scaled2 = data.matrix(x_scaled_train[,-c(4,6)]) 
# design_scaled2 is the design matrix of the 7 scaled predictor variables. 
Correlation_matrix2 = round(t(design_scaled2)%*%(design_scaled2),2)
Correlation_matrix2 #Pairwise correleation between the variables is low to moderate.
det(Correlation_matrix2)
# Determinant value is 0.2952089. Multi-collinearity may or may not be present.

model2=lm(y_scaled_train~.,data=red_train_x)#Reduced model with intercept.
summary(model2)
# Computing VIF
round(vif(model2),2) #Computing VIF using the reduced model with intercept
##All the VIF values are less than 5, so multi-collinearity is not present.

#VERIFICATION OF ASSUMPTIONS

#Test for Normality
ols_plot_resid_qq(reduced_model) #Assumption of normality is not satisfied.

#Shapiro test cannot be performed because sample size is not between 3 and 5000.
#So we conduct Lilliefors test.
library(nortest)
lillie.test(reduced_model$residuals)##p-value<0.05=>normality assumption is violated

library(robustbase)
rob=lmrob(y_scaled_train~.-1,data=red_train_x)
summary(rob)
coeff_robust= as.matrix(rob$coefficients)
X_test=as.matrix(red_test_x)
y_pred_robust=X_test%*%coeff_robust
rmse_robust=rmse(y_scaled_test,y_pred_robust) #0.01756416
##Plotting the predicted and observed values
library(lattice)
library(latex2exp)

xyplot(y_scaled_test[1:100] + y_pred_robust[1:100] ~ 1:100,
       data = data,
       type = c("l"),
       ylab = "Response & Predicted Values",
       xlab = "Index",
       main = list(
         "Plot of Predicted and observed Values from Robust Regression",
         cex = 1.2
       ),
       auto.key = list(
         space = "top",
         columns = 2,
         text = c(
           "Response Values",
           "Predicted Values"
         )
       ))

#Detecting Heteroscedasticity
plot(reduced_model,1,lwd=3)
library(skedastic)
glejser(mainlm=reduced_model,sigma="main",statonly=FALSE)
#p value <0.05 => Heteroscedasticity is present in the dataset.

#Weighted Least Squares
cal.weights <- 1 / lm(abs(reduced_model$residuals) ~ reduced_model$fitted.values)$fitted.values^2
wls=lm(y_scaled_train~.-1, data =red_train_x,weights = cal.weights)
summary(wls)
#Adj R2=0.9862, RSE=1.283

wls_pred=predict(wls,red_test_x)
rmse(y_scaled_test,wls_pred)# RMSE=0.01800191
##Plotting the predicted and observed values
library(lattice)
library(latex2exp)

xyplot(y_scaled_test[1:100] + wls_pred[1:100] ~ 1:100,
       data = data,
       type = c("l"),
       ylab = "Response & Predicted Values",
       xlab = "Index",
       main = list(
         "Plot of Predicted and observed Values from WLS Regression",
         cex = 1.2
       ),
       auto.key = list(
         space = "top",
         columns = 2,
         text = c(
           "Response Values",
           "Predicted Values"
         )
       ))

#Test for autocorrelation
red_model_res=reduced_model$residuals
acf(red_model_res, main="ACF Plot for residuals") #Presence of autocorrelation
pacf(red_model_res, main="PACF Plot for residuals") #Presence of autocorrelation

#Runs test
library(randtests)
runs.test(red_model_res,plot=TRUE)
#p-value<0.05
#Autocorrelation is present.

#Durbin-Watson test
durbinWatsonTest(model2)
#DW statistic=0.5452369. 
#Lag=1=> Residuals follow AR(1) process.
#Autocorrelation is present.

#Applying Orcutt-Cochrane procedure to remove autocorrelation
library(orcutt)
auto=lm(y_scaled_train~.,data=red_train_x)
coch=cochrane.orcutt(auto)
summary(coch)#RSE=0.0056, pvalue<<<<0.05
auto.pred=predict(coch,red_test_x)
rmse(y_scaled_test,auto.pred)# RMSE=0.024

##Plotting the predicted and observed values
library(lattice)
library(latex2exp)

xyplot(y_scaled_test[1:100] + auto.pred[1:100] ~ 1:100,
       data = data,
       type = c("l"),
       ylab = "Response & Predicted Values",
       xlab = "Index",
       main = list(
         "Plot of Predicted and observed Values after removing autocorrelation",
         cex = 1.2
       ),
       auto.key = list(
         space = "top",
         columns = 2,
         text = c(
           "Response Values",
           "Predicted Values"
         )
       ))