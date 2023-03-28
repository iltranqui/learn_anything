%Evaluate Jump in time series
 clc,
 clear all,
 close all,
 format short
 input=xlsread('Example 5','Sheet1','C:C');
 %% Partitioning and stationarity, MannWhitney test (Cardillo's)
 if bitget(numel(input),1) % for odd number
 sample = input(1:end-1);
 samples= reshape(sample,[numel(sample)/2 2]);
 mwSTATS=mwwtest(samples(:,1)',samples(:,2)');
 else %for even number
 samples = reshape(input,[numel(input)/2 2]);
 mwSTATS=mwwtest(samples(:,1)',samples(:,2)');
 end
 if mwSTATS.p(1,2) < 0.01
 disp(' significant JUMP is detected in the series')
 else
 disp(' significant JUMP is NOT detected in the series')
 end

%Evaluate Jump in time series
 clc,
 clear all,
 close all,
 format short
 input=xlsread('Example 5','Sheet1','C:C');
 %% Partitioning and stationarity, MannWhitney test (ranksum)
 sample1 = input(1:round(numel(input)/2), 1);
sample2 = input(numel(sample1)+1:end,1);
 [pVal H stat]=ranksum(sample1,sample2, 'alpha' , 0.01, 'method' , … 
 fprintf ('p-value = %d \nTest rejection decisions(H) = %d\nRankSum 
stat = %d\n',...
 pVal, H, stat.ranksum)
 if pVal < 0.01
 disp(' significant JUMP is detected in the series')
 else
 disp(' significant JUMP is NOT detected in the series')
end