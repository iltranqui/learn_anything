clear all;
close all;
clc;

%%  Uncertain model of the Mass/Damper/Spring system
%
m = ureal('m',3,'Percentage',10);
c = ureal('c',1,'Percentage',10);
k = ureal('k',2,'Percentage',10);
%
u = icsignal(1);
x = icsignal(1);
xdot = icsignal(1);
M = iconnect;
M.Input = u;
M.Output = x;
M.Equation{1} = equate(x,tf(1,[1,0])*xdot);
M.Equation{2} = equate(xdot,tf(1/m,[1,0])*(u-k*x-c*xdot));
G = M.System;

%% Weigthing Functions
tol = 0.1;
% Sensitivity Function desired
nuWS = [0.001 2];
dnWS = [0.05 1];
gainWS = 5e0;
WS = gainWS*tf(nuWS,dnWS);
% Complementary Sensitivity Function desired
nuWT = [0.05 1];
dnWT = [0.0001 1];
gainWT = 5e-2;
WT = gainWT*tf(nuWT,dnWT);
% Control Sensitivity Function desired
nuWK0 = [0.05 1];
dnWK0 = [0.001 1];
gainWK0 = 5e-2;
WK0 = gainWK0*tf(nuWK0,dnWK0);

% Plot of WS, WK0 and WT
figure(1)
bode(WS,'b-',WK0,'r--',WT,'g-.',WS+WT,'cyan.'),
grid
legend('WS','WK0','WT','WS+WT')

%% H-inf synthesis

[K_h,cl_h,gam,info] = mixsyn(G,WS,WK0,WT);

%% Analysis of the H-inf synthesis for all Wegihthing Functions 

looptransfer = loopsens(G.Nominal,K_h);
L = looptransfer.Lo;
T = looptransfer.To;
S = looptransfer.So;
K0 = looptransfer.So*K_h;
I = eye(size(L));
figure(2)
omega = logspace(-1,3,100);
sigma(S,'b-',gam/WS,'b-.',T,'r-',gam/WT,'r-.',K0,'cyan-',gam/WK0,'cyan-.',omega)
grid
legend('\sigma(S) performance', ...
'\sigma(1/WS) robustness bound',...
'\sigma(T) performance', ...
'\sigma(1/WT) robustness bound',...
'\sigma(K0) performance ', ...
'\sigma(1/WK0) robustness bound')

figure(3)
omega = logspace(-1,3,100);
sigma(L,'b-',WS/gam,'r--',gam/WT,'r.',omega)
grid
legend('\sigma(L)','\sigma(WS) performance bound', ...
'\sigma(1/WT) robustness bound')