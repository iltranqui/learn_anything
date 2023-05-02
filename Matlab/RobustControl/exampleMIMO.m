clear all
close all
clc

% system is MIMO. 2 input and 2 output
% to each input and uncertainty is added
%
% 
%   G(nom) =   
% 
%%

s = tf('s');
g11 = 12/(0.2*s + 1);
g12 = -0.05/(0.1*s + 1);
g21 = 0.1/(0.3*s + 1);
g22 = 5/(0.7*s - 1);
Gnom = [g11 g12;g21 g22];
w1 = makeweight(0.1,20,10);
w2 = makeweight(0.2,25,10);
W = blkdiag(w1,w2);
Delta_1 = ultidyn('Delta_1',[1 1]);
Delta_2 = ultidyn('Delta_2',[1 1]);
Delta = blkdiag(Delta_1,Delta_2);
G = (eye(2) + Delta*W)*Gnom;

optsbode = bodeoptions;
optsbode.MagUnits = 'abs';

%% plot G nominal
figure(1)
bode(Gnom,'b-',G,'r-.')
grid
legend('Gnom','G')
title('Nominal and uncertain system')
xlabel('Frequency (rad/s)')
ylabel('Magnitude (dB)');

% The uncertainty in the first output is 10 % in the low frequency range -> value is 0.1
% increases to 100 % at ω = 20 rad/s and reaches 1000 % in the high frequency range. -> value is 1 and 10 respectively
% The uncertainty in the second output is 20 % in the low frequency range, 100 % at -> value is 0.2 and 1 respectively
% ω = 25 rad/s and 1000 % in the high frequency range. -> value is 10
%% plot w1 and w2 
figure(2)
bode(w1,'b-',w2,'r-.',optsbode)
grid
legend('w1','w2')
title('Weighting functions')
xlabel('Frequency (rad/s)')
ylabel('Magnitude (dB)');

% The plant singular values are shown in Fig. 11.3. It is seen that after the frequency of
% 20 rad/s there is a significant uncertainty in the plant which may reach up to 20 dB
% at frequencies larger than 100 rad/s. Hence, the controller should be designed so that
% the open-loop system has gain that is smaller than −20 dB for ω > 100 rad/s.
% plot singular values of G
plotoptions = sigmaoptions;  
plotoptions.Grid = 'on';
omega = logspace(-1,2,100);
figure(3)
sigma(Gnom,'b-',omega,plotoptions)
hold on;
sigma(G,'r-.',omega,plotoptions)
grid
legend('Gnom','G')
title('Singular values of nominal and uncertain system')
xlabel('Frequency (rad/s)')
ylabel('Magnitude (dB)');

% ----------------- 1.2 -----------------
% 
% Goals: 
%Robustness requirements: Roll-off −20 dB/decade and a gain of −20 dB at
% frequency 100 rad/s,
% Performance requirements: Maximize 1/σ(S) in the high frequency range.

% Both requirements may be satisfied taking the desired open-loop transfer function
% matrix as
% Gd (s) = 10/s

Gd = 10/s;
[K,cls,gam] = loopsyn(Gnom,Gd);

looptransfer = loopsens(Gnom,K);
L = looptransfer.Lo;
figure(4)
sigma(L,'r-',Gd,'b--',Gd/gam,'k-.',Gd*gam,'k-.',omega,plotoptions)
grid
legend('Open Loop','Gd','Gd/gam','Gd*gam')
title('Singular values of L and Gd')
xlabel('Frequency (rad/s)')
ylabel('Magnitude (dB)');

% mixsyn 
% The weighting function W1 is used to shape the sensitivity function in the low frequency
W1 = (s + 10)/(2*s + 0.3);
W3 = (s + 10)/(0.05*s + 20);
[K_h,cl_h,gam] = mixsyn(Gnom,W1,[],W3);

figure(5)
bode(W1,'b-',W2,'r-.',optsbode)
grid
legend('w1','w2')
title('Weighting functions')
xlabel('Frequency (rad/s)')
ylabel('Magnitude (dB)');