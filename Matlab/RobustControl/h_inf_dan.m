%% Computation of the transfer function of the disturbance of lame d'aria
% then placing a hinf controller to reject a distrubance by slowing adding elements

n = [0,0.00125651358244363,0.485202372848273,0.164792809408070,36.7224968577778,5.42803475694088,721.425823282285,43.1652240656235,3498.93941028222];
d = [1,0.441820758123519,111.335782541048,36.3843645886072,3740.96166435174,805.039021066104,43075.1527319086,4590.65750896580,136872.539479771];

G= tf([n],[d]);
G.InputName = 'u(force)';  
G.OutputName = 'y(position)';

% Bode plot of the transfer function
figure(1)
bodemag(G)
grid on

% disturbance transfer function for lame d'aria and measurement
Wact = 0.8*tf([1 50],[1 500]);  Wact.u = 'u(dist)';  Wact.y = 'e1';  % Actuator force with a weight of 0.8 and a low-pass filter with a corner frequency of 50 rad/s
Wmes = ss(0.5); Wmes.u = 'd(meas)';   Wmes.y = 'Wd3';  % Disturbance on the measurement with a constant weight of 0.5

actuator  = sumblk('y1 = y(position)+e1)');
measurement = sumblk('y2 = y1+Wd3');

ICinputs = {'u(force)';'u(dist)';'d(meas)'};
ICoutputs = {'y2','y(position)','y1'};
qcaric = connect(G,Wact,Wmes,actuator,measurement,ICinputs,ICoutputs);

% plot all 3 transfer functions of system qcaric 
% Open Loop
figure(2)
bodemag(qcaric)
grid on

% Controller 
K = tf([1 0.1],[1 0.01]); K.u = 'error1'; K.y = 'u(force)';  % Controller with a low-pass filter with a corner frequency of 0.1 rad/s

control = sumblk('error1 = y2');

% Closed-loop models
CL = connect(qcaric,K, ...
    {'u(dist)';'d(meas)'}, ...
    {'y(position)','y1','y2'});
