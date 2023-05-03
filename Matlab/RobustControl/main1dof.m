clear all;
close all;
clc;

% Building the Model
% mod_mds

m = 3;
c = 1;
k = 2;
pm = 0.4;
pc = 0.2;
pk = 0.3;
%
A=[ 0 1
-k/m -c/m];

B1 = [  0      0    0
        -pm -pc/m -pk/m];

B2 = [ 0  1/m];

C1 = [ -k/m -c/m
        0    c
        k    0 ];

C2 =  [ 1    0 ];

D11 = [-pm -pc/m -pk/m
        0    0    0   
        0    0    0 ];

D12 = [1/m  0   0];

D21 = [0 0 0];

D22 = 0;

% G = pck(A,[B1,B2],[C1;C2],[D11 D12;D21 D22])

% Building the uncertain model
% LFT representation of the mass-damper-spring system with uncertainties
% sys_mds
m_nom = 3; c_nom = 1; k_nom = 2;
p_m = 0.4; p_c = 0.2; p_k = 0.3;
mat_mi = [-p_m 1/m_nom; -p_m 1/m_nom];
mat_c = [0 c_nom; p_c c_nom];
mat_k = [0 k_nom; p_k k_nom];
int1 = nd2sys([1],[1 0]);
int2 = nd2sys([1],[1 0]);

systemnames = 'mat_mi mat_c mat_k int1 int2';
sysoutname = 'G';
inputvar = '[um;uc;uk;u]';
input_to_mat_mi = '[um;u-mat_c(2)-mat_k(2)]';
input_to_mat_c = '[uc;int1]';
input_to_mat_k = '[uk;int2]';
input_to_int1 = '[mat_mi(2)]';
input_to_int2 = '[int1]';
outputvar = '[mat_mi(1);mat_c(1);mat_k(1);int2]';
sysic;

% From mod_mds .> somehow
% Uncertain model of the Mass/Damper/Spring system
%
m = ureal('m',3.5,'Percentage',40);
c = ureal('c',2,'Percentage',40);
k = ureal('k',2,'Percentage',40);

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

% G is the system in 1dof

% pfr_mds
% Frequency responses of the perturbed plants

omega = logspace(-1,1,100);
G64 = gridureal(G,'c',4,'k',4,'m',4);

% Bode plot of the Open Loop unedited Function
figure(1)
bode(G.Nominal,'r-',G64,'b--',omega), grid
title('Bode plots of uncertain plant')
legend('Nominal plant','Uncertain plant')

% wts mds.m updated alos with v2013
% the desired behaviour of the function, the desired plant, even with the uncertainties 
% has to behave like this plant. Hinf will be doing exactly this
Time = 1; % sec
smorz = 0.8; % smorzamento
nuM = 1;
dnM = [Time^2 Time*smorz*1.0 1];
gainM = 1;
M = gainM*tf(nuM,dnM)
tol = 1e-6;

% Comparison of the original nominal plant and the desired plant
figure(2)
bode(G.Nominal,'r--',M,'b-x',G64,'k:.',omega), grid
title('Bode plots of uncertain plant')
legend('Nominal plant','Desired plant','Uncertain plant')

% --------------------------------------------------------------------------
%% Building the Sensitivity Functions for mixsyn
% --------------------------------------------------------------------------
% thoguh i don'0t have yet the controller, i can build the sensitivity functions
% for the desired plant, and then i will use them to build the controller
% Sensitivity Function desired
nuWS = [1];
dnWS = [0.03 1];
gainWS = 1e1;
WS = gainWS*tf(nuWS,dnWS)
% This weighting function has the purpose to ensure the gain of
%the loop from r and d to the error y − yM to be of order tol in the low frequency
%range which will ensure closeness between the system and model and sufficient
%disturbance attenuation at the system output. 

% Complementary Sensitivity Function desired
nuWt = [0.04 1];
dnWt = [0.002 1];
gainWt = 5e-1;
Wt = gainWt*tf(nuWt,dnWt)
% This weighting function ensures attenuation of componetns with frequency over 10rad/s

% Control Sensitivity Function desired
nuWKS = [0.05 1];
dnWKS = [0.001 1];
gainWKS = 5e-2;
Wks = gainWKS*tf(nuWKS,dnWKS)

% Plot the weighting functions
figure(3)
%subplot(2, 2, 1)
optsbode = bodeoptions;
optsbode.MagUnits = 'abs';
bodeplot(WS,'r-',Wt,'g--',Wks,'k--',optsbode), 
grid on;
title('Weighting functions')
legend('WS','WT','Wks')

% --------------------------------------------------------------------------
%% Inverse weighthing Functiosn
% --------------------------------------------------------------------------
nuM = 1;
dnM = [1.0^2 2*0.7*1.0 1];
gainM = 1.0;
M = gainM*tf(nuM,dnM);
tol = 10^(-2);
nuWp = [2 1];
dnWp = [2 tol];
gainWp = 5*10^(-1);
Wp = gainWp*tf(nuWp,dnWp);
nuWu = [0.05 1];
dnWu = [0.0001 1];
gainWu = 5.0*10^(-2);
Wu = gainWu*tf(nuWu,dnWu);

% plotting 
% Plot the weighting functions
figure(4)
%subplot(2, 2, 1)
optsbode = bodeoptions;
optsbode.MagUnits = 'abs';
bodeplot(Wp,'r-',Wu,'g--',optsbode), 
grid on;
title('Weighting functions')
legend('Wp','Wu')



% --------------------------------------------------------------------------

%% building the opne model of the uncertain mass-damper-spring
% olp_mds
% figure 8.8
% open-loop connection with the weighting function
% How to write an uncertain system
systemnames = ' G M Wp Wu ';              % the names of the TF to insert
inputvar = '[ ref; dist; control ]';         % the names of the input TFs
outputvar = '[ Wp; -Wu; ref-G-dist ]';     % the names of the output TFs
input_to_G = '[ control ]';
input_to_M =  '[ ref ]';             % Since G has only 1 input 
input_to_Wp = '[ G+dist-M ]';             % only 1 input
input_to_Wu = '[ control ]';            % only 1 input
sys_ic_init = sysic;
%
% open-loop connection with the weighting function
% How to write an uncertain system
systemnames = ' G M WS Wt ';              % the names of the TF to insert
inputvar = '[ ref; dist; control ]';         % the names of the input TFs
outputvar = '[ WS; -Wt; ref-G-dist ]';     % the names of the output TFs
input_to_G = '[ control ]';
input_to_M =  '[ ref ]';             % Since G has only 1 input 
input_to_Wp = '[ G+dist-M ]';             % only 1 input
input_to_Wu = '[ control ]';            % only 1 input
sys_ic = sysic;


%% H - inf synthesis of the controller
% hin_mds
nmeas = 1;
ncon = 1;
gmin = 0.1;
gmax = 10;
tol = 0.001;
[K_hin,clp,gam] = hinfsyn(sys_ic_init.NominalValue,nmeas,ncon,[gmin,gmax])

% Bode of the Controller
figure
bode(K_hin,'b-'), grid on;
title('Bode plots of controller')
legend('K_hin')

%% Robust stability analysis of the closed loop
clp_ic= lft(sys_ic,K_hin)
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
%figure
%semilogx(info.MussvBnds(1,1),'r-',info.MussvBnds(1,2),'b--')












%% New System to obtain the Sensitivity Functions and the Closed Loop
systemnames = ' G ';
inputvar = '[ ref; dist; control ]';         % the names of the input TFs
outputvar = '[ G+dist; control; ref-G-dist ]';     % the names of the output TFs
input_to_G2 = '[ control ]';
sim_ic = sysic;
cls = lft(sim_ic,K_hin);

% Transfer Functions obtained via 
T0 = cls(1,1);
S0 = cls(1,2);
KS0 = -cls(2,2);

% plot all the TFs
figure
bode(To,'r-',S0,'b--',KS0,'g--'), grid on;
title('Bode plots of uncertain plant')
legend('To','S0','KS0')

% Plot the weighting functions with the Target ->z Wrong ! S
figure
bode(1/Wp,'r.',1/Wu,'b.',M,'g.'), 
grid on;
hold on;
title('Weighting functions with target Sensitivity TF')
bode(T0,'r-',S0,'b--',KS0,'g--'), grid on;
legend('Wp','Wu','M','To','S0','KS0')

% Obtain responses of the uncertrain systems
figure
T64 = gridureal(T0,'c',4,'k',4,'m',4);
bode(M,'r-',T64,'b--',omega), grid on;
title('Bode plots of uncertain plant')
legend('M','T0')

% Comparison between sensitivity of the uncertain system and inverse performance weigting function
% All the Dynamic Systems must be below the 1/Wp
figure
omega = logspace(-3,2,100);
S64 = gridureal(S0,'c',4,'k',4,'m',4);
bode(1/Wp,'r-',S64,'b--',omega), grid on;
title('Bode plots of the Sensitivity TF')
legend('1/Wp','S0') 

% Comparison between Complementary sensitivity of the uncertain system and inverse performance weigting function
% all the Dynamic Systems must be below the 1/Wu
figure
omega = logspace(-3,2,100);
S64 = gridureal(T0,'c',4,'k',4,'m',4);
bode(1/Wu,'r-',S64,'b--',omega), grid on;
title('Bode plots of the Complementary Sensitivity TF')
legend('1/Wu','KS0')

% Comparison between Control sensitivity of the uncertain system and inverse performance weigting function
figure
omega = logspace(-3,2,100);
KS64 = gridureal(KS0,'c',4,'k',4,'m',4);
bode(KS64,'b--',omega), grid on;
title('Bode plots of the Complementary Sensitivity TF')
legend('KS0')