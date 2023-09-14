clear all;
close all;
clc;

% syste, 36_35 node
n_36 =[0.00962850008217881,	0.576109900602084,	0.476627495334119,	19.2560317365358	,5.94623868114044,	146.993292909239,	16.8159353248908	,226.960880128106];
d_36 = [1,	0.422387101891881	,51.6766246603535	,16.2503969695907	,735.680566168432	,153.524424660450,	3127.51205589723	,326.519550574759	,2975.63431566161];

% system 38_35 node
n_38 = [0.00125651358244363,0.485202372848273,0.164792809408070,	36.7224968577778,	5.42803475694088	,721.425823282285	,43.1652240656235,	3498.93941028222];
d_38 = [1,0.441820758123519,111.335782541048,36.3843645886072	,3740.96166435174,	805.039021066104	,43075.1527319086	,4590.65750896580,	136872.539479771];

G= tf([n_36],[d_36]);
G.InputName = 'u_{force}';  
G.OutputName = 'y_{position}';

D = tf([n_38],[d_38]);
D.InputName = 'u_{lamearia}';
D.OutputName = 'y_{lamearia}';

% Wegiht functions
s=tf('s');
W_error = ( 0.5*s+0.015)/(s+0.00015);
W_control = ( s+100)/(s+200);
W3_output = [];

P = augw(G,W_error,W_control,W3_output);
P.InputName = {'reference','u_{input}'};
P.OutputName = {'y_{error}','y_{input}','y_{output}'};

% Controller
ncont = 1;
nmeas = 1;
[K] = hinfsyn(P,nmeas,ncont);

% bode plots of P 
figure(1)
bode(P)
grid on
title('Bode plot of P')

K_tf = tf(K);

% cell to vector
[K_num,K_den] = tfdata(K_tf);
K_num = cell2mat(K_num);
K_den = cell2mat(K_den);

[G_num,G_den] = tfdata(G);
G_num = cell2mat(G_num);
G_den = cell2mat(G_den);

F = feedback(G*K_tf,1);

% bode plots of F
figure(2)
bode(F)
hold on
grid on
bode(G)
title('Bode plot of F')
legend('F','G')

Aug = feedback(K_tf*P(2,:),eye(2));

% bode plots of Aug
figure(3)
bode(Aug)
grid on
title('Bode plot of Aug')


