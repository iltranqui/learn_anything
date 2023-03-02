%Evaluate Periodicity in time series
 clc,
 clear all,
 close all,
format long 
 input=xlsread('Example 6','Sheet1','B:B');
 %% Fisger's g test (Shteingart)
 %calculating periodogram for N/2 of data
 [Pxx,F,w] = periodogram(input,rectwin(length(input)),length(input));
 Pxx = Pxx(2:length(input)/2);
 %finding the significant peak and g-statistic
 [maxval,index] = max(Pxx);
 fisher_g = Pxx(index)/sum(Pxx);
 %calculating the P-value of the g-statistic based on Wichert et al. 
% (2004)
 F = F(2:end-1);
 F_sig=F(index);
 N = length(Pxx);
 for nn = 1:3
 I(nn) = (-1)^(nn-1)*nchoosek(N,nn)*(1-nn*fisher_g)^(N-1);
 end
 pval = sum(I);
 fprintf ('Fisher g-statistic= %d\nP-value= %d\nSignificant Frequnecy= 
',fisher_g,pval,F_sig)
 if pval < 0.05
 disp(' significant Periodicity is detected in the series')
else
 disp(' significant Periodicity is NOT detected in the series')
 end