% Applying the MIXsyn techiques with specific performance goals to a system composed
% of 1 dof ! 

clear all;
close all;
clc;

%%  Uncertain model of the Mass/Damper/Spring system
%
smorz = ureal('smorz',0.1,'Percentage',10);   % 0<smorz<1
wn = ureal('wn',10,'Percentage',10);   % in rad/s
%
u = icsignal(1);
x = icsignal(1);
xdot = icsignal(1);
M = iconnect;
M.Input = u;
M.Output = x;
M.Equation{1} = equate(x,tf(1,[1,0])*xdot);
M.Equation{2} = equate(xdot,tf(1,[1,0])*(u-wn^2*x-smorz*wn*xdot));
G = M.System;

% Plot Bode
figure(1)
bode(G,'b-',G.Nominal,'r--')
grid
legend('G','G.Nominal')

