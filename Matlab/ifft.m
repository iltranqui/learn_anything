close all; 
clear all;

% Define the time step and duration of the signal
dt = 0.01; % Time step
t = 0:dt:2; % Time vector

% Define the frequency range of the signal
fmin = 0.1; % Minimum frequency
fmax = 2; % Maximum frequency

% Generate the time signal
x = chirp(t,fmin,1,fmax);

% Compute the Fourier transform of the signal
X = fft(x);

% Compute the frequency vector
f = (0:length(x)-1)/(dt*length(x));

% Compute the inverse Fourier transform of the signal
x_rec = ifft(X);

% Plot the signal in time domain
figure
plot(t,x)
title('Time signal')
xlabel('Time (s)')

% Plot the signal in frequency domain
figure
plot(f,abs(X))
title('Frequency signal')
xlabel('Frequency (Hz)')

% Plot the original signal and the reconstructed signal
figure
plot(t,x,'b')
hold on
plot(t,x_rec,'r--')
legend('Original signal','Reconstructed signal')
title('Comparison of original signal and reconstructed signal')
xlabel('Time (s)')

% This will give you a graph with two lines, one line in blue representing the original signal,
% and the other line in red representing the signal obtained via the IFFT.