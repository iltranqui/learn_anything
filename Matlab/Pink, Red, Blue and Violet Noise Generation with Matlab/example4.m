clear, clc, close all

% signal parameters
fs = 44100;     % sampling frequency, Hz         
M = 256;        % number of the rows of the noise matrix
N = 256;        % number of the columns of the noise matrix

% noise generation
% Note: here one could learn how to generate a  
% m-by-n matrix with noise samples (columnwise). 
xred = arrayfun(@(x) rednoise(M), M*ones(1, N), 'UniformOutput', false); 
xred = cell2mat(xred);
xpink = arrayfun(@(x) pinknoise(M), M*ones(1, N), 'UniformOutput', false); 
xpink = cell2mat(xpink);
xblue = arrayfun(@(x) bluenoise(M), M*ones(1, N), 'UniformOutput', false); 
xblue = cell2mat(xblue);
xviolet = arrayfun(@(x) violetnoise(M), M*ones(1, N), 'UniformOutput', false); 
xviolet = cell2mat(xviolet);

% visual presentation
figure(1)
colormap gray
subplot(2, 2, 1)
imagesc(xred)
title('Red noise')
subplot(2, 2, 2)
imagesc(xpink)
title('Pink noise')
subplot(2, 2, 3)
imagesc(xblue)
title('Blue noise')
subplot(2, 2, 4)
imagesc(xviolet)
title('Violet noise')