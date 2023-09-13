clear all;
close all;
clc;

% syste, 36_35 node
n_36 =[0.00962850008217881,	0.576109900602084,	0.476627495334119,	19.2560317365358	,5.94623868114044,	146.993292909239,	16.8159353248908	,226.960880128106];
d_36 = [1,	0.422387101891881	,51.6766246603535	,16.2503969695907	,735.680566168432	,153.524424660450,	3127.51205589723	,326.519550574759	,2975.63431566161];

% system 38_35 node
n_38 = [0.00125651358244363	,0.485202372848273	,0.164792809408070,	36.7224968577778,	5.42803475694088	,721.425823282285	,43.1652240656235,	3498.93941028222];
d_38 = [1,	0.441820758123519	,111.335782541048,	36.3843645886072	,3740.96166435174,	805.039021066104	,43075.1527319086	,4590.65750896580,	136872.539479771];

G= tf([n_36],[d_36]);
G.InputName = 'u_{force}';  
G.OutputName = 'y_{position}';

D = tf([n_38],[d_38]);
D.InputName = 'u_{lamearia}';
D.OutputName = 'y_{lamearia}';

% disturbance transfer function for lame d'aria and measurement
%Wmes = ss(0.5); Wmes.u = 'd(meas)';   Wmes.y = 'Wd3';  % Disturbance on the measurement with a constant weight of 0.5

actuator  = sumblk('y_{post_dist} = y_{position}+y_{lamearia}');
%measurement = sumblk('y_{postmes} = y_{post_dist}+Wd3');

%ICinputs = {'u_{force}';'u_{lamearia}';'d(meas)'};
ICinputs = {'u_{force}';'u_{lamearia}'};
%ICoutputs = {'y_{postmes}','y_{position}','y_{post_dist}'};
ICoutputs = {'y_{position}','y_{post_dist}'};
%qcaric = connect(G,D,Wmes,actuator,measurement,ICinputs,ICoutputs);
qcaric_onlydist = connect(G,D,actuator,ICinputs,ICoutputs);

% plot all 3 transfer functions of system qcaric 
% Open Loop
figure(2)
bodemag(qcaric_onlydist)
title('Bode plot of Open Loop TF qcaric(s)')
grid on

% Final desired closed-loop targets for the gain loops
Disturbance_atten = 0.04 * tf([1/8 1],[1/80 1]);     % Handling target target function  closed-loop targets
Position_atten = 0.4 * tf([1/0.45 1],[1/150 1]);   % Comfort target target function   closed-loop targets
Disturbance_atten.u = 'y_{post_dist}'; Disturbance_atten.y = 'y_{Handling}';
Position_atten.u = 'y_{position}'; Position_atten.y = 'y_{Comfort}';

% Connection to the weighting functions
ICinputs = {'u_{force}';'u_{lamearia}'};
ICoutputs = {'y_{position}','y_{post_dist}','y_{Handling}','y_{Comfort}'};
qcaric_dist = connect(G,D,actuator,Disturbance_atten,Position_atten,ICinputs,ICoutputs);

% plot all transfer functions of system qcaric_dist 
figure(3)
bodemag(qcaric_dist)
title('Bode plot of Open Loop TF qcaric(s)')
grid on

qcaric_dist_hinf = qcaric_dist(3:4,:);

ncont = 1; % one control signal, u
nmeas = 1; % two measurement signals, sd and ab
[K,~,gamma] = hinfsyn(qcaric_dist_hinf,nmeas,ncont);


% connect in Closed Loop
% Closed-loop models
K.u = {'-y_{post_dist}'};  K.y = 'u_{force}';
ICinputs_CL = {'u_{lamearia}'};
ICoutputs_CL = {'y_{position}','y_{post_dist}','y_{Handling}','y_{Comfort}'};
CL = connect(qcaric_dist,K,ICinputs_CL,ICoutputs_CL);

