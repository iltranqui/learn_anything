% Principles of System Identification: Theory and Practice page 725
close all;
clear all;
%% Create process object
dgp_arx = idpoly ([1 -0.9 0.2] ,[0.3 0.2] ,1 ,1 ,1);
dgp_arx.iodelay = 3;    

%% Low-frequency input design
% pseudo-random binary sequence (PRBS) input to excite the process.
% Syntax is such that [0 1] is a full band input
uk = idinput(2044,'prbs',[0 1/3] ,[ -1 1]);
% Simulation
xk = sim(dgp_arx ,uk);
mult_fac = (1.2*(1 - 0.2^2) - 0.9^2 + 0.9^2*0.2) /(1 + 0.2);
dgp_arx. Noisevariance = mult_fac*var(xk)/10;
yk = sim(dgp_arx ,uk , simOptions ('AddNoise',true));
% result in a white-noise variance of σ^2e = 0.0657

Zk = iddata(yk,uk,1);
%% Partition the data into training and test sets
Ztr = Zk(1:1500); Ztest = Zk(1501:end);

%Partition the data into training and test sets
Ztr = Zk (1:1500); Ztest = Zk (1501:end);
% Detrend data
[Ztrd ,Tr] = detrend(Ztr ,0);
% Align the test data w.r.t. t
% % Detrend data
% [Ztrd] = detrend(Ztr,0);
% %  Plot the data
% figure;
% plot(Ztrd)
% hold on; 
% plot(Ztr)
% Tl = Ztr - Ztrd; %Trend line
% 
% mu_tr = mean(Ztr);
% sigma_tr = std(Ztr);

% Align test data to have the same mean and standard deviation as training data
% Ztest_aligned = (Ztest - mean(Ztest)) * sigma_tr / std(Ztest) + mu_tr;
% Align the test data w.r.t. the same mean and variance
%Ztestd = detrend(Ztrd,Tl);
% figure;
% plot(Ztest);
% hold on;
% plot(Ztest_aligned)
Ztestd = detrend(Ztrd ,Tr);

%% NON-PARAMETRIC ANALYSIS
%Our next step is to estimate impulse, step and frequency response models because they reveal useful
%information on delay and process dynamics. The only significant assumption that we make here is
%that the system is LTI.

% IR estimation
fir_mod = impulseest (Ztrd ,20);
figure; impulseplot (fir_mod ,'sd',3)
% Step response estimation
figure; step(fir_mod);

% FRF estimation
[Gwhat ,wvec] = spa(Ztrd);
figure; bode(Gwhat);

