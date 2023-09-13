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
unc = ultidyn('unc',[1 1],'SampleStateDim',50);
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
% Can this be approximated to a white noise ? 

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
% Selecting only the 3rd and 2nd output of the qcar model
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

% The three controllers achieve closed-loop H∞ norms of 0.94, 0.67 and 0.89, respectively. 
% Construct the corresponding closed-loop models and compare the gains from road disturbance to x_b
% s_d and a_b for the passive and active suspensions.
% Observe that all three controllers reduce suspension deflection and body acceleration below 
% the rattlespace frequency (23 rad/s).

% Closed-loop models
K.u = {'sd(suspension_travel)','ab(body_acceleration)'};  K.y = 'u';
CL = connect(qcar,Act.Nominal,K,'r(dist)',{'xb(car_position)';'sd(suspension_travel)';'ab(body_acceleration)'});

figure(6)
bodemag(qcar(:,'r(dist)'),'b', CL(:,:,1),'r-.', ...
   CL(:,:,2),'m-.', CL(:,:,3),'k-.',{1,140}), grid
legend('Open-loop','Comfort','Balanced','Handling','location','SouthEast')
title('Body travel, suspension deflection, and body acceleration due to road')





%% Time Doman Simulation
% To further evaluate the three designs, perform time-domain simulations using a road disturbance signal r(t) representing a road bump of height 5 cm.
% Road disturbance

t = 0:0.0025:1;
roaddist = zeros(size(t));
roaddist(1:101) = 0.025*(1-cos(8*pi*t(1:101)));

% Closed-loop model
SIMK = connect(qcar,Act.Nominal,K,'r(dist)',{'xb(car_position)';'sd(suspension_travel)';'ab(body_acceleration)';'fs(act)'});

% Simulate
p1 = lsim(qcar(:,1),roaddist,t);
y1 = lsim(SIMK(1:4,1,1),roaddist,t);
y2 = lsim(SIMK(1:4,1,2),roaddist,t);
y3 = lsim(SIMK(1:4,1,3),roaddist,t);


% Plot results
figure(7);
subplot(211)
plot(t,p1(:,1),'b',t,y1(:,1),'r.',t,y2(:,1),'m.',t,y3(:,1),'k.',t,roaddist,'g')
grid on;
title('Body travel'), ylabel('x_b (m)')
subplot(212)
plot(t,p1(:,3),'b',t,y1(:,3),'r.',t,y2(:,3),'m.',t,y3(:,3),'k.',t,roaddist,'g')
title('Body acceleration'), ylabel('a_b (m/s^2)')

subplot(211)
plot(t,p1(:,2),'b',t,y1(:,2),'r.',t,y2(:,2),'m.',t,y3(:,2),'k.',t,roaddist,'g')
title('Suspension deflection'), xlabel('Time (s)'), ylabel('s_d (m)')
subplot(212)
plot(t,zeros(size(t)),'b',t,y1(:,4),'r.',t,y2(:,4),'m.',t,y3(:,4),'k.',t,roaddist,'g')
title('Control force'), xlabel('Time (s)'), ylabel('f_s (kN)')
legend('Open-loop','Comfort','Balanced','Handling','Road Disturbance','location','SouthEast')



%% Robust Mu Design
% So far you have designed H∞ controllers that meet the performance objectives for the nominal actuator model. 
% Next use μ-synthesis to design a controller that achieves robust performance for the entire family of actuator models. The robust controller is synthesized with the mu-syn 
% function using the uncertain model qcaric(:,:,2) corresponding to "balanced" performance (β=0.5).

[Krob,rpMU] = musyn(qcaric(:,:,2),nmeas,ncont)

% Closed-loop model (nominal)
Krob.u = {'sd(suspension_travel)','ab(body_acceleration)'};
Krob.y = 'u';
SIMKrob = connect(qcar,Act.Nominal,Krob,'r(dist)',{'xb(car_position)';'sd(suspension_travel)';'ab(body_acceleration)';'fs(act)'});

% Simulate the nominal response to a road bump with the robust controller Krob. 
% The responses are similar to those obtained with the "balanced" H∞ controller.

% Simulate
p1 = lsim(qcar(:,1),roaddist,t);
y1 = lsim(SIMKrob(1:4,1),roaddist,t);

% Plot results
figure(8)
clf, subplot(221)
plot(t,p1(:,1),'b',t,y1(:,1),'r',t,roaddist,'g')
title('Body travel'), ylabel('x_b (m)')
subplot(222)
plot(t,p1(:,3),'b',t,y1(:,3),'r')
title('Body acceleration'), ylabel('a_b (m/s^2)')
subplot(223)
plot(t,p1(:,2),'b',t,y1(:,2),'r')
title('Suspension deflection'), xlabel('Time (s)'), ylabel('s_d (m)')
subplot(224)
plot(t,zeros(size(t)),'b',t,y1(:,4),'r')
title('Control force'), xlabel('Time (s)'), ylabel('f_s (kN)')
legend('Open-loop','Robust design','location','SouthEast')

% Next simulate the response to a road bump for 100 actuator models randomly selected from the uncertain model set Act.
rng('default'), nsamp = 100;  clf

% Uncertain closed-loop model with balanced H-infinity controller
figure(9)
CLU = connect(qcar,Act,K(:,:,2),'r(dist)',{'xb(car_position)';'sd(suspension_travel)';'ab(body_acceleration)'});
lsim(usample(CLU,nsamp),'b',CLU.Nominal,'r',roaddist,t)
title('Nominal "balanced" design')
legend('Perturbed','Nominal','location','SouthEast')

% Uncertain closed-loop model with balanced robust controller
figure(10)
CLU = connect(qcar,Act,Krob,'r(dist)',{'xb(car_position)';'sd(suspension_travel)';'ab(body_acceleration)'});
lsim(usample(CLU,nsamp),'b',CLU.Nominal,'r',roaddist,t)
title('Robust "balanced" design')
legend('Perturbed','Nominal','location','SouthEast')

% The robust controller Krob reduces variability due to model uncertainty and delivers more consistent performance.



%% Controller Simplification: Order Reduction

%The robust controller Krob has relatively high order compared to the plant.
%  You can use the model reduction functions to find a lower-order controller that achieves the same level of robust performance.
% Use reduce to generate approximations of various orders.

% Create array of reduced-order controllers
NS = order(Krob);
StateOrders = 1:NS;
Kred = reduce(Krob,StateOrders);



% Closed-loop model (nominal)
Krob.u = {'sd(suspension_travel)';'ab(body_acceleration)'};
Krob.y = 'u';
SIMKrob = connect(qcar,Act.Nominal,Krob,'r(dist)',{'xb(car_position)';'sd(suspension_travel)';'ab(body_acceleration)'});

% Compute robust performance margin for each reduced controller
gamma = 1;
CLP = lft(qcaric(:,:,2),Kred);
for k=1:NS
   PM(k) = robgain(CLP(:,:,k),gamma);
end

% Compare robust performance of reduced- and full-order controllers
PMfull = PM(end).LowerBound;
plot(StateOrders,[PM.LowerBound],'b-o',...
   StateOrders,repmat(PMfull,[1 NS]),'r');
grid
title('Robust performance margin as a function of controller order')
legend('Reduced order','Full order','location','SouthEast')


%% Controller Simplification: Fixed-Order Tuning
% Alternatively, you can use musyn to directly tune low-order controllers.
%  This is often more effective than a-posteriori reduction of the full-order controller Krob. 
% For example, tune a third-order controller to optimize its robust performance.

% Create tunable 3rd-order controller 
K = tunableSS('K',3,ncont,nmeas);

% Tune robust performance of closed-loop system CL
CL0 = lft(qcaric(:,:,2),K);
[CL,RP] = musyn(CL0);


K3 = getBlockValue(CL,'K');
bode(K3)