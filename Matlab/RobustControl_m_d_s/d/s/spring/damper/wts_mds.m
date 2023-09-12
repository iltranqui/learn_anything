% Performance weighting functions for the mass/damper/spring
% system
% The weighting functions Wp and Wu are used to reflect the relative
% significance of the performance requirement over different frequency ranges.
% For the mass/damper/spring system, the weighting functions Wp and Wu are
% used to reflect the relative significance of the performance requirement
% over different frequency ranges for different functions:

% Sensitivity Function
% Complementary Sensitivity Function
% Control Sensitivity Function

%clear all;
close all;
clc;

nuWp = [1  1.8  10  ];
dnWp = [1  8    0.01];
gainWp = 0.95;
Wp = gainWp*idtf(nuWp,dnWp)
%
nuWu = 1;
dnWu = 1;
gainWu = 10^(-2);
Wu = gainWu*idtf(nuWu,dnWu)

% Plot the weighting functions
figure
bodemag(Wp,'r-',Wu,'b--'), 
grid on;
title('Weighting functions')
legend('Wp','Wu')

% Inverse weighthing Function
% to achieve the desired performance of disturbance rejection (or, of tracking
%error) it is necessary to satisfy the inequality |Wp(I + GK)−1|∞ < 1. Since
%Wp is a scalar function in the present case, the singular values of the sensitivity
%function (I +GK)^−1 over the frequency range must lie below that of 1/wp .
figure
omega = logspace(-3,3,1000);
bodemag(1/Wp,'r-',omega), grid
title('Inverse of Performance Weighting Function')

% from frequency 1 rad/s the distrubance is no longer attenuated
