%% 
% To create a stochastic model (ARMA, ARIMA, SARIMA) , the built-in arima function can be used

%% Stochastic model

models = arima(
'ARLags',p,   %  Vector of positive integer lags associated with nonseasonal autoregressive
'MALags',q, ...  % Vector of positive integer lags associated with nonseasonal MA coefficients
'SARLags',P,    % Vector of positive integer lags associated with seasonal autoregressive coefficients
'SMALags',Q, ...  % Vector of positive integer lags associated with seasonal MA coefficients
'D', d,   % Nonnegative integer indicating the degree of the n onseasonal differencing lag
‘Seasonality', w, ...  % Nonnegative integer indicating the order of the seasonal differencing lag operator
'constant', C,   % Scalar constant in the linear time series
'Distribution','Gaussian');  % Conditional probability distribution of the innovation process, can also be t-student

% Example
%% Stochastic model 
 models = arima('ARLags',1:2,'MALags',1,...
'SARLags',12:12:24,'SMALags',12:12:36,...
 'D',0,'Seasonality',12) 


% Stochastic model  Example with t-student
 models_t_student = arima('ARLags',1:2,'MALags',1,...
'SARLags',12:12:24,'SMALags',12:12:36,...
 'D',0,'Seasonality',12); 

 tdist = struct('Name','t','DoF',10); 
 models.Distribution = tdist

% ARIMA model with no seasonality
 models = arima('ARLags',p,'MALags',q,...
 'D',d,'constant', c,...
 'Distribution','Gaussian'); 

% ARIMA model wiht specific lags
 models = arima('ARLags',[1 2 12],'MALags',1,...
 'D',1,'constant', 0,...
 'Distribution','Gaussian')

%% Polynomial parameter estimation
% Estimating the parameters 
 [EstMdl,EstParamCov,logL,info] = estimate(models,y) 

% Applied Example
Create the SARIMA model and Estimate parameters 
 clc, 
 clear all, 
 close all, 
 format short
 %we import the normalized time series 
 input = xlsread('Example 4.xlsx', 'sheet1', 'c:c'); 
8 %partitioning for model training 
 y = input(1:144); 
 %% Creating models 
 %creating the SARIMA(0,1,1)(1,0,0)12 
 model1 = arima('MALags',1, 'SARLags',12, 'D',1); 
 %shorthand syntax of ARIMA(1,1,1) 
 model2 = arima(1,1,1); 
 %% Estimating the parameters 
 [EstMdl1,EstParamCov1,logL1,info1] = estimate(model1,y); 
 [EstMdl2,EstParamCov2,logL2,info2] = estimate(model2,y); 

% Following the modeling exercise, we can see that each model
% produces a different estimation. Determination of the most suitable model will depend
% on computation of statistical indices as well at the fitness criteria

%% Infering the Residuals 
% Residulas of the models 
 [Resi1,V1] = infer(EstMdl1,y); 
 [Resi2,V2] = infer(EstMdl2,y); 
% modeled series 
 y_hat1 = y - Resi1; 
 y_hat2 = y - Resi2; 
 N_series = iddata(y, 'TimeUnit', 'months'); 
 m1 = iddata(y_hat1, 'TimeUnit', 'months'); 
 m2 =iddata(y_hat2, 'TimeUnit', 'months'); 
 figure 
 compare(N_series, m1, '--r', m2) 
 ylabel('Normalized Precipitation (mm)'); 
 legend('Normalized precipitation series',...
 'SARIMA (0,1,1)(1,0,0)^{12}',...
 'ARIMA (1,1,1)',...
 'FontSize',10);
 legend('boxoff'); 

 % To estimate and
% forecast the stochastic models in MATLAB software, there must be sufficient presample
%responses to initialize the autoregressive terms and it must have enough innovations
%to initialize the MA terms. 

%% Stochastic model 
 models = arima('AR',{value1,…,value_p},...
 'MA', {value1,…,value_q},...
 'SAR',{value1,…,value_P},...
'SMA', {value1,…,value_Q},...
 'D',d,'Seasonality',w,...
 'constant', values_c,...
 'Distribution','Gaussian'); 
 %% Estimating the parameters 
 EstModels = estimate(models, y,'Y0', Y0,...
 'AR0', value_p0,'MA0', value_q0,)


%% Creating models 
% Create the SARIMA model and Estimate parameters 
clc, 
 clear all, 
 close all, 
 format short
 %we imprt the normalized time series 
 input = xlsread('Example 5.xlsx', 'sheet1', 'c:c'); 
 %partitioning for model training 
 y_0 = input(1:72); %first half
 y = input(73:144); %second half
 %% Creating models 
 %creatin
 %creating the SARIMA(2,1,1)(1,1,1)12 
 model1 = arima('AR', {-1.04, -0.48},'MA', {0.22},...
 'SAR',{-0.17},'SMA',{0.22},...
 'D',1, 'Seasonality', 12,...
 'Constant', 0.07, 'Variance', 10.7); 
 %shorthand syntax of ARIMA(2,1,1) 
 model2 = arima (2,1,1); 
 %% Estimating initials 
 [EstMdl0,EstParamCov0,logL0,info0] = estimate(model2,y_0); 
 %% Estimate final parameters of model 2 
 con0 = EstMdl0.Constant; 
 ar0 = {EstMdl0.AR{1}, EstMdl0.AR{2}}; 
 ma0 = EstMdl0.MA{1}; 
 var0 = EstMdl0.Variance; 
 [EstMdl2,EstParamCov,logL,info] = estimate(model2,y,...
 'AR0',ar0,'MA0',ma0,...
 'Constant0',con0,'Variance0',var0);
 %% Residulas of the models 
 [Resi1,V1] = infer(model1,y); 
 [Resi2,V2] = infer(EstMdl2,y); 
 %% modeled series 
 y_hat1 = y - Resi1; 
 y_hat2 = y - Resi2; 
 N_series = iddata(y, 'TimeUnit', 'months'); 
 m1 = iddata(y_hat1, 'TimeUnit', 'months'); 
 m2 =iddata(y_hat2, 'TimeUnit', 'months'); 
 figure 
 compare(N_series, m1, '--r', m2) 
 ylabel('Normalized Precipitation (mm)'); 
 legend('Normalized precipitation series',...
 'SARIMA (2,1,1)(1,1,1)^{12}',...
 'ARIMA (2,1,1)',...
 'FontSize',10);
 legend('boxoff'); 



%% Optimization methods in stochstic methods
% MATLAB software has the ability of implementing optimization methods for estimating
%the model parameters.To consider optimization,the estimate function should be altered
%to include the ‘Options’ input argument followed by the name of the optimization
% method to be considered For example, if we wanted to constrain the model tolerance to
% 1 × 10−6, then the syntax is as follows:

Otions = optimoptions(@fmincon,'ConstraintTolerance',1e-6,'Algorithm','sqp');

% Estimate final parameters of model 3 (ARIMA without initials) 
 Options = optimoptions(@fmincon,'ConstraintTolerance',...
 1e-6,'Algorithm','sqp'); 
 EstMdl = estimate(model,y, 'Options' , Options ); 

% However, there are many optimization methods that are built-in to MATLAB and are available for use ( list to insert slowly )

%% Stochastic models with exogenous inputs
%  These models use a
% single time series as input to the models to arrive at the desired output values when
% forecasting. stochastic models can consider additional input variables in order
% to model (StochasticX) and forecast the desired output 
% These extra inputs are termed exogenous time series, and their parameter values are
% determined outside of the model their linear effect is imposed on the current model.

% The stochastic models using exogenous series (stochasticX) have the same stationarity
% requirements as the univariate models

%% Stochastic model 
 models = arima('AR',{value1,…,value_p},'MA', {value1,…,value_q},...
 'SAR',{value1,…,value_P},...
 'SMA', {value1,…,value_Q},...
 'D',d,'Seasonality',w,...
 'Beat',B , constant', values_c,...
 'Distribution','Gaussian'); 
 %% Estimating the parameters 
 EstModels = estimate(models, y, 'X', Exs,'Y0',Y0,...
 'AR0', value_p0,'MA0', value_q0,...
 'SAR0', value_P0, 'SMA0', value_Q0,...
 'Beat0', value_B0, Variance0' = V0); 

 %% %% Estimate ARX Model Parameters Using Initial Values 
 clc, 
 clear all, 
 close all, 
 format short
 load Data_CreditDefaults
 Exs = Data(:,[1 3:4]); %defining exogenous series AGE, CPF and SPR 
 T = size(Exs,1); %size of exogenous matrix
 y = Data(:,5); %IGD series
 %% Partitioning the time series 
 y0 = y(1); %pre-sample
 yEst = y(2:T); 
 XEst = Exs(2:end,:); 
 Beta0 = [0.5 0.5 0.5]; %each for one exogenous series
 %% creating the AR(1) model 
 models = arima('AR', -0.017311 , 'Constant', -0.20477 ); 
 %% Fit the model to the data and specify the initial values. 
 EstMdl = estimate(models,yEst,'X',XEst,...
 'Y0',y0,'Beta0',Bet)





