% Applying the MIXsyn techiques with specific performance goals to a system composed
% of 1 dof ! 

clear all;
close all;
clc;

%%  Uncertain model of the Mass/Damper/Spring system
%
smorz = ureal('smorz',0.01,'Percentage',10);   % 0<smorz<1
wn = ureal('wn',0.1759*(2*pi),'Percentage',10);   % in rad/s

smorz_1 = ureal('smorz_1',0.01,'Percentage',10);   % 0<smorz_1<1
wn_1 = ureal('wn_1',0.36*(2*pi),'Percentage',10);   % in rad/s

smorz_2 = ureal('smorz_2',0.01,'Percentage',10);   % 0<smorz_2<1
wn_2 = ureal('wn_2',0.595*(2*pi),'Percentage',10);   % in rad/s

smorz_3 = ureal('smorz_3',0.01,'Percentage',10);   % 0<smorz_3<1
wn_3 = ureal('wn_3',0.889*(2*pi),'Percentage',10);   % in rad/s
%

gain = 1e1;
gain_2 = 1e-1;

u = icsignal(1);
x1 = icsignal(1);
xdot1 = icsignal(1);
x2 = icsignal(1);
xdot2 = icsignal(1);
x3 = icsignal(1);
xdot3 = icsignal(1);
x4 = icsignal(1);
xdot4 = icsignal(1);

M = iconnect;
M.Input = u;
M.Output = gain_2*(x1+x2+x3+x4);
M.Equation{1} = equate(x1,tf(1,[1,0])*xdot1);
M.Equation{2} = equate(xdot1,tf(1*gain,[1,0])*(u-wn^2*x1-smorz*wn*xdot1));
M.Equation{3} = equate(x2,tf(1,[1,0])*xdot2);
M.Equation{4} = equate(xdot2,tf(1*gain,[1,0])*(u-wn_1^2*x2-smorz_1*wn_1*xdot2));
M.Equation{5} = equate(x3,tf(1,[1,0])*xdot3);
M.Equation{6} = equate(xdot3,tf(1*gain,[1,0])*(u-wn_2^2*x3-smorz_2*wn_2*xdot3));
M.Equation{7} = equate(x4,tf(1,[1,0])*xdot4);
M.Equation{8} = equate(xdot4,tf(1*gain,[1,0])*(u-wn_3^2*x4-smorz_3*wn_3*xdot4));
G = M.System;

load("sys_36_35.mat")
opts = bodeoptions;
opts.FreqUnits = 'Hz';
% Plot Bode
figure(1)
%bodeplot(sys_36_35,opts)
hold on
bodeplot(G,'b-',G.Nominal,'r--',opts)
grid
legend('G','G.Nominal')

%% Weigthing Functions
tol = 0.1;
% Sensitivity Function desired
nuWS = [0.001 2];
dnWS = [0.05 1];
gainWS = 5e-1;
WS = gainWS*tf(nuWS,dnWS);
% Complementary Sensitivity Function desired
nuWT = [0.05 1];
dnWT = [0.0001 1];
gainWT = 5e-2;
WT = gainWT*tf(nuWT,dnWT);
% Control Sensitivity Function desired
nuWK0 = conv([0.001 1],[0.001 1]);
dnWK0 = conv([0.0001 1],[0.0001 1]);
gainWK0 = 5e-2;
WK0 = gainWK0*tf(nuWK0,dnWK0);

% Plot of WS, WK0 and WT
figure(2)
bode(WS,'b-',WK0,'r--',WT,'g-.',WS+WT,'cyan.'),
grid
legend('WS','WK0','WT','WS+WT')

%% H-inf synthesis

opts = hinfsynOptions;
opts.Method = 'LMI';
opts.Display = 'on';
opts.TolRS = 0.3;

[K_h,cl_h,gam,info] = mixsyn(G,WS,WK0,WT);

%% Analysis of the H-inf synthesis for all Wegihthing Functions 

looptransfer = loopsens(G.Nominal,K_h);
L = looptransfer.Lo;
T = looptransfer.To;
S = looptransfer.So;
K0 = looptransfer.So*K_h;
I = eye(size(L));
figure(3)
omega = logspace(-1,3,1000);
sigma(S,'b-',gam/WS,'b.',T,'r-',gam/WT,'r.',K0,'cyan-',gam/WK0,'cyan.',omega)
grid
legend('\sigma(S) performance', ...
'\sigma(1/WS) robustness bound',...
'\sigma(T) performance', ...
'\sigma(1/WT) robustness bound',...
'\sigma(K0) performance ', ...
'\sigma(1/WK0) robustness bound')

%% PLot the resulting closed Loop
Loop = series(K_h,G);
Cl = feedback(Loop, 1);
omega = logspace(-1,3,1000);
figure(4)
bode(G.Nominal,'b-',Loop,'r-',Cl,'g-.',omega)
grid
legend('G (Plant)','L (OpenLoop)','Cl (Closed Loop)')

%% Robust stability analysis of the closed loop
clp_ic= lft(G,K_h)
omega = logspace(-2,2,100);
clp_g = ufrd(clp_ic,omega);
opt = robopt('Display','on');
[stabmarg,destabu,report,info]=robuststab(clp_g,opt)
report
figure
%semilogx(info.MussvBnds(1,1),'r-',info.MussvBnds(1,2),'b--')
%legend

% Robust performance analysis of the closed loop
opt = robopt('Display','on');
[stabmarg,destabu,report,info]=robustperf(clp_g,opt)
report
