% Find and interpolating missing data in time series
clc,
clear all,
close all,
format long
%Read data from source (Excel file with *.xlsx format)
input = readtable('AirQualityUCI.xlsx', 'Sheet', "Sheet1", 'Range', 
"A:C",'TreatAsEmpty',{'.','NA'});
%Summarize the input
 summary(input)
 %Find Rows with Missing Values
 Missed = ismissing(input,{'' '.' 'NA' 'N/A' NaN NaT -99});
 rowsWithMissing = input(any(Missed,2),:);
 disp(rowsWithMissing)
%Replace Missing Value with MATLAB standard indicator 'NaN'
 input = standardizeMissing(input,{'' '.' 'NA' 'N/A' NaN NaT -99});
 disp(input)
 summary(input)
 %% First option of treating missing data
 % Remove missing data
 [output1 output2F]= rmmissing(input);
 disp(output1)
 summary(output1)
%% Second option of treating missing data
%Interpolating missing data with linear method
 [output2,output2F] = fillmissing(input,'linear'); % or spline or pchip nearest
 % can used other methods like pchip,spline, makia and other methofs to fill in the data
[output2,output2F] = fillmissing(input,'linear'); % or spline or pchip nearest
 rowsInrerpolated = output2(any(output2F,2),:);
 disp(rowsInrerpolated)
 %Plot interpolated data
 output2FF = output2F(:,3);
 plot(input.t, input.DATA,'d', input.t(output2FF),output2.DATA(output2FF),'o')
 xlabel('Time (any time interval)', 'fontsize',12)
 ylabel('Collected data', 'fontsize',12)
 legend('Observed','Interpolated','Location','southeast')