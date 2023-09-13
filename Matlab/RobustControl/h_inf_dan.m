%% Computation of the transfer function of the disturbance of lame d'aria
% then placing a hinf controller to reject a distrubance by slowing adding elements

close all
clear all
clc

n = [0,0.00125651358244363,0.485202372848273,0.164792809408070,36.7224968577778,5.42803475694088,721.425823282285,43.1652240656235,3498.93941028222];
d = [1,0.441820758123519,111.335782541048,36.3843645886072,3740.96166435174,805.039021066104,43075.1527319086,4590.65750896580,136872.539479771];

% syste, 36_35 node
n_36 =[0.00962850008217881,	0.576109900602084,	0.476627495334119,	19.2560317365358	,5.94623868114044,	146.993292909239,	16.8159353248908	,226.960880128106];
d_36 = [1,	0.422387101891881	,51.6766246603535	,16.2503969695907	,735.680566168432	,153.524424660450,	3127.51205589723	,326.519550574759	,2975.63431566161];

% system 38_35 node
n_38 = [0.00125651358244363	,0.485202372848273	,0.164792809408070,	36.7224968577778,	5.42803475694088	,721.425823282285	,43.1652240656235,	3498.93941028222];
d_38 = [1,	0.441820758123519	,111.335782541048,	36.3843645886072	,3740.96166435174,	805.039021066104	,43075.1527319086	,4590.65750896580,	136872.539479771];

G= tf([n],[d]);
G.InputName = 'u(force)';  
G.OutputName = 'y_{position}';

% Bode plot of the transfer function
figure(1)
bodemag(G)
title('Bode plot of the transfer function G(s)')
grid on

% disturbance transfer function for lame d'aria and measurement
Wact = 0.8*tf([1 50],[1 500]);  Wact.u = 'u(dist)';  Wact.y = 'e1';  % Actuator force with a weight of 0.8 and a low-pass filter with a corner frequency of 50 rad/s
Wmes = ss(0.5); Wmes.u = 'd(meas)';   Wmes.y = 'Wd3';  % Disturbance on the measurement with a constant weight of 0.5

actuator  = sumblk('y_{post_dist} = y_{position}+e1');
measurement = sumblk('y_{postmes} = y_{post_dist}+Wd3');

ICinputs = {'u(force)';'u(dist)';'d(meas)'};
ICoutputs = {'y_{position}','y_{postmes}','y_{post_dist}'};
qcaric = connect(G,Wact,Wmes,actuator,measurement,ICinputs,ICoutputs);

% plot all 3 transfer functions of system qcaric 
% Open Loop
figure(2)
bodemag(qcaric)
title('Bode plot of Open Loop TF qcaric(s)')
grid on

%% Controller 
K = tf([1 0.1],[1 0.01]); K.u = 'error1'; K.y = 'u(force)';  % Controller with a low-pass filter with a corner frequency of 0.1 rad/s

control = sumblk('error1 = -y_{postmes}');

% Closed-loop models
CL = connect(qcaric,K,control, ...
    {'u(dist)';'d(meas)'}, ...
    {'y_{position}','y_{post_dist}','y_{postmes}'});

% plot all 6 transfer functions of system CL
figure(3)
bodemag(CL)
title('Bode plot of Closed Loop TF CL(s)')
grid on

%% Hinf controller
% Final desired closed-loop targets for the gain loops
Disturbance_atten = 0.04 * tf([1/8 1],[1/80 1]);     % Handling target target function  closed-loop targets
Measure_atten = 0.4 * tf([1/0.45 1],[1/150 1]);   % Comfort target target function   closed-loop targets
Disturbance_atten.u = 'y_{post_dist}'; Disturbance_atten.y = 'y_{Handling}';
Measure_atten.u = 'y_{postmes}'; Measure_atten.y = 'y_{Comfort}';


% Closed Loop with weitghing functions
CL_hinf = connect(qcaric(2:3,:),K,control,Disturbance_atten,Measure_atten, ...
    {'u(dist)';'d(meas)'}, ...
    {'y_{post_dist}','y_{postmes}','y_{Handling}','y_{Comfort}'});

% Hinf controller
ncont = 1; % one control signal, u
nmeas = 2; % two measurement signals, sd and ab
[K] = mixsyn(CL_hinf,ncont,nmeas);

% executes but have i done ? 