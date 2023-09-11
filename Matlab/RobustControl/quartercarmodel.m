% https://it.mathworks.com/help/robust/gs/active-suspension-control-design.html

%% Physical parameters of a 1/4 quarter car 
close all;
clear all;
clc;

% Physical parameters
mb = 300;    % kg
mw = 60;     % kg
bs = 1000;   % N/m/s
ks = 16000 ; % N/m
kt = 190000; % N/m

% State matrices
A = [ 0 1 0 0; [-ks -bs ks bs]/mb ; ...
      0 0 0 1; [ks bs -ks-kt -bs]/mw];
B = [ 0 0; 0 1e3/mb ; 0 0 ; [kt -1e3]/mw];
C = [1 0 0 0; 1 0 -1 0; A(2,:)];
D = [0 0; 0 0; B(2,:)];

% State space model
qcar = ss(A,B,C,D);
qcar.StateName = {'body travel (m)';'body vel (m/s)';...
          'wheel travel (m)';'wheel vel (m/s)'};   % State names
qcar.InputName = {'r(dist)';'fs(act)'};  % Input names
qcar.OutputName = {'xb(car_position)';'sd(suspension_travel)';'ab(body_acceleration)'};  % Output names

% Frequency response of the quarter car model
figure(1)
bodemag(qcar({'ab(body_acceleration)','sd(suspension_travel)'},'r(dist)'),'b',qcar({'ab(body_acceleration)','sd(suspension_travel)'},'fs(act)'),'r',{1 100});   % Bode plot of the tranfer function from r to ab and sd
legend('Road disturbance (r)','Actuator force (fs)','location','SouthWest')  % Add legend
title({'Gain from road dist (r) and actuator force (fs) ';    % ... and title 
       'to body accel (ab) and suspension travel (sd)'})


% Plot all the bodemag plots from every input to every output in a 3x2 table with title and legend for each plot
figure(2)
bodemag(qcar,'b',{1 1000}), grid
title('Frequency response of the quarter car model')


% The hydraulic actuator used for active suspension control is connected between the body mass mb
% and the wheel assembly mass mw . The nominal actuator dynamics are represented by the first-order 
% transfer function 1/(1+s/60) with a maximum displacement of 0.05 m.

ActNom = tf(1,[1/60 1]);

% This nominal model only approximates the physical actuator dynamics. We can use a family of actuator models to account 
% for modeling errors and variability in the actuator and quarter-car models. This family consists of a nominal model with a frequency-dependent amount of uncertainty.

Wunc = makeweight(0.40,15,3);
unc = ultidyn('unc',[1 1],'SampleStateDim',5);
Act = ActNom*(1 + Wunc*unc);
Act.InputName = 'u';
Act.OutputName = 'fs(act)';

% Actuator uncertainty
figure(3)
rng('default')
bode(Act,'b',Act.NominalValue,'r+',logspace(-1,3,120))
 
%% H-inf design for disturbance rejection


Wroad = ss(0.07);  Wroad.u = 'd1';   Wroad.y = 'r(dist)';   % Road disturbance with a constant weight of 0.07
Wact = 0.8*tf([1 50],[1 500]);  Wact.u = 'u';  Wact.y = 'e1';  % Actuator force with a weight of 0.8 and a low-pass filter with a corner frequency of 50 rad/s
Wd2 = ss(0.01);  Wd2.u = 'd2';   Wd2.y = 'Wd2';  % Disturbance on the measurement cnstant weight of 0.01
Wd3 = ss(0.5);   Wd3.u = 'd3';   Wd3.y = 'Wd3';  % Disturbance on the measurement with a constant weight of 0.5

% Specify closed-loop targets for the gain from road disturbance r to suspension deflection s_d (handling)
% and body acceleration a_d b(comfort). 
% Because of the actuator uncertainty and imaginary-axis zeros, only seek to attenuate disturbances below 10 rad/s.

% Final desired closed-loop targets for the gain loops
HandlingTarget = 0.04 * tf([1/8 1],[1/80 1]);     % Handling target target function  closed-loop targets
ComfortTarget = 0.4 * tf([1/0.45 1],[1/150 1]);   % Comfort target target function   closed-loop targets

Targets = [HandlingTarget ; ComfortTarget];
figure(4)
bodemag(qcar({'sd(suspension_travel)','ab(body_acceleration)'},'r(dist)')*Wroad,'b',Targets,'r--',{1,1000}), grid
title('Response to road disturbance')
legend('Open-loop currently','Closed-loop target')

% To investigate the trade-off between passenger comfort and road handling, construct three sets of weights (βW_sd,(1−β)W
% ab) corresponding to three different trade-offs: comfort (β=0.01), balanced (β=0.5), and handling (β=0.99).

% Three design points
beta = reshape([0.01 0.5 0.99],[1 1 3]);
Wsd = beta / HandlingTarget;
Wsd.u = 'sd(suspension_travel)';  Wsd.y = 'e3';
Wab = (1-beta) / ComfortTarget;
Wab.u = 'ab(body_acceleration)';  Wab.y = 'e2';

% plot the three design targets with all the bodepoints 
% legend for each plot ( there are 6 plots in total)
figure(5)
bodemag(Wsd,'g',Wab,'m',{1,1000}), grid
title('Response to different betas')
legend('beta = 0.01','beta = 0.5','beta = 0.99','beta = 0.01','beta = 0.5','beta = 0.99')

%% Construct model
% use connect to construct a model qcaric of the block diagram of Figure 2. 
% Note that qcaric is an array of three models, one for each design point β.
% Also, qcaric is an uncertain model since it contains the uncertain actuator model Act.



sdmeas  = sumblk('y1 = sd(suspension_travel)+Wd2');
abmeas = sumblk('y2 = ab(body_acceleration)+Wd3');
ICinputs = {'d1';'d2';'d3';'u'};
ICoutputs = {'e1';'e2';'e3';'y1';'y2'};
qcaric = connect(qcar(2:3,:),Act,Wroad,Wact,Wab,Wsd,Wd2,Wd3,...
                 sdmeas,abmeas,ICinputs,ICoutputs)

                 

%% Nominal H-infinity Design
ncont = 1; % one control signal, u
nmeas = 2; % two measurement signals, sd and ab
K = ss(zeros(ncont,nmeas,3));
gamma = zeros(3,1);
for i=1:3
   [K(:,:,i),~,gamma(i)] = hinfsyn(qcaric(:,:,i),nmeas,ncont);
end

gamma