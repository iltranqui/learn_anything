clear, clc, close all

% signal parameters
fs = 44100;         % sampling frequency, Hz
T = 5;              % signal duration, s
N = round(fs*T);    % number of samples

% noise generation
xred = rednoise(N);         % red (Brownian) noise
xpink = pinknoise(N);       % pink (flicker) noise
xblue = bluenoise(N);       % blue noise
xviolet = violetnoise(N);   % violet noise

% sound presentation
soundsc(xred, fs)
pause(T+1)

soundsc(xpink, fs)
pause(T+1)

soundsc(xblue, fs)
pause(T+1)

soundsc(xviolet, fs)