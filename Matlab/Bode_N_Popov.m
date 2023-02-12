%% import the TF as TF or State Space as SS
close all; 
clear all; 

% TF as minimum polynomial
% f(x) = s(s-7)(s-6) + (s+4)(s-9)(s-2) + s(s-8)(s+1) 
rts1 = [ 0  7  6];
rts1_d = [7 6 0 ];
rts2 = [-4  9  2];
rts3 = [ 0  8 -1];
trm1 = poly(rts1);
trm1_d = poly(rts1_d);
trm2 = poly(rts2);
trm3 = poly(rts3);
den = trm1 + trm2 + trm3;
den_d = trm1_d + trm2 + trm3;

a1 = 1;
a2 = 5;
a3 = 10;
% a1*s^2 + a2*s + a3
num = [a1 a2 a3]; 

A = [0 1 0;
     0 0 1;
     -1 -1 -5]
B=[0 0 1]'
C=[1 0 0]
D=[0]

%% num explain as an expanded vector
% 
% TF
 G_tf = tf(num,den)
 G_tf_d = tf(num,den_d)
% SS
 G_ss = ss(A,B,C,D)
 [n,d]=ss2tf(A,B,C,D);
 G=tf(n,d)

 %% draw Bode plot 
 figure;
 bode(G_tf);
 hold on;
 bode(G_ss);

 %% Draw Nyquist 
 figure;
 nyquist(G_tf);
 hold on;
 nyquist(G_ss);

 %% PLot Popov PLot 
popov(G_ss,0,0)
