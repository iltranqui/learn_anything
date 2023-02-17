%% IMport the dataset

%Read data from source (Excel file with *.xlsx format)
input = readtable('AirQualityUCI.xlsx','Sheet',"Sheet1",
'Range',"A:C",'TreatAsEmpty',{'.','NA'});
%Summarize the input
summary(input)

%Find Rows with Missing Values
 Missed = ismissing(input,{'' '.' 'NA' 'N/A' NaN NaT -99});
 rowsWithMissing = input(any(Missed,2),:);
 disp(rowsWithMissing)

% Replace Missing Value with MATLAB standard indicator 'NaN' 
 input = standardizeMissing(input,{'' '.' 'NA' 'N/A' NaN NaT -99}); 
 disp(input) 
 summary(input) 

% Remove missing data
[output1 output2F]= rmmissing(input);
disp(output1)

%Interpolating missing data with linear method
 [output,outputF] = fillmissing(input,'linear'); 
 rowsInrerpolated = output(any(outputF,2),:);
 disp(rowsInrerpolated)

 