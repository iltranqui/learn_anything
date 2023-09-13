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

Position_atten = 0.4 * tf([1/0.45 1],[1/150 1]);   % Comfort target target function   closed-loop targets
Position_atten.u = 'y_{position}'; Position_atten.y = 'y_{positionatten}';

ICinputs = {'u_{force}'};  % Input channel names for initial conditions
ICoutputs = {'y_{position}','y_{positionatten}'}; % Output channel names for initial conditions
model = connect(G,Position_atten,ICinputs,ICoutputs);

%% Nominal H-infinity Design
ncont = 1; % one control signal, u
nmeas = 1; % two measurement signals, sd and ab
K = hinfsyn(model,ncont,nmeas);

%% Closed-loop system
ref = ss(1); ref.u = 'ref'; ref.y = 'ref_y';
reference = sumblk('error = ref_y-y_{position}');
% Closed-loop models
K.u = {'error'};  K.y = 'u_{force}';  % Closed-loop connections
CL = connect(model,K,ref,{'ref'},{'y_{position}','y_{positionatten}'});