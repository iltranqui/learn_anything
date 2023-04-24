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

% plot G 
omega = logspace(-2,2,100);
figure()
hold on;
sigma(G,'r-',Gnom,'b--x',omega)
title('Bode plots of G')
grid on;
legend('G','Gnom')

% Plot the Bode Plots of G 
figure;
bodeplot(G,'r-',Gnom,'b--x');
grid
legend('G','Gnom')

% plot the Sensitivity Functions



% The plant singular values are shown in Fig. 11.3. It is seen that after the frequency of
% 20 rad/s there is a significant uncertainty in the plant which may reach up to 20 dB
% at frequencies larger than 100 rad/s. Hence, the controller should be designed so that
% the open-loop system has gain that is smaller than −20 dB for ω > 100 rad/s.

Gd = 10/s;
[K,cls,gam] = loopsyn(Gnom,Gd);

looptransfer = loopsens(Gnom,K);
omega = logspace(-1,2,100);
L = looptransfer.Lo;
sigma(L,'r-',Gd,'b--',Gd/gam,'k-.',Gd*gam,'k-.',omega)
grid on;
title('Singular Value Plot of L')
legend('\sigma(L) loopshape','\sigma(Gd) desired loop','\sigma(Gd) + gam','\sigma(Gd) - gam')

T = looptransfer.To;
% Robust stability analysis of the closed loop
omega = logspace(-1,3,100);
I = eye(size(L));
sigma(I+L,'r-',T,'b--',omega);
grid on;
hold on;
sigma(1/w1,'-',1/w2,'cyan--',omega);
legend('1/\sigma(S) performance','\sigma(T) robustness','weight1','Weight2')

% Plot the Bode Plot of the Sensitivity Functions
S = looptransfer.So;
T = looptransfer.To;
Ks = looptransfer.So*K;

figure
omega = logspace(-1,3,100);
bodeplot(S,'r-',T,'b--',Ks,'cyan-.',omega);
grid
legend('Sensitivty performance','Complementary Sensitivty robustness','Control Sensitivity')
title('Bode Sensitivity Functions')

%% Singulare Values 
% Still have to understand them
S = looptransfer.So;
T = looptransfer.To;
KS = looptransfer.So*K;
% Robust stability analysis of the closed loop
omega = logspace(-1,3,100);
sigma(S,'r-',T,'b--',omega);
grid on;
hold on;
sigma(1/w1,'-',1/w2,'cyan--',omega);
legend('1/\sigma(S) performance','\sigma(T) robustness','weight1','Weight2')

%% Example: 
W1 = (s + 10)/(2*s + 0.3);
W3 = (s + 10)/(0.05*s + 20);
[K_h,cl_h,gam] = mixsyn(Gnom,W1,[],W3);

% This figure
%show the singular values of S and T and the performance and robustness bounds W1 and 1/W3, respectively. It is seen that the minimum singular
%value of S−1 lies below the magnitude response of W1 and the maximum singular
%value of T is below the magnitude response of 1/W3. This means that the performance and robustness requirements specified by the weighting functions W1 and
% W3, are satisfied.
figure
looptransfer = loopsens(Gnom,K_h);
L = looptransfer.Lo;
T = looptransfer.To;
I = eye(size(L));
figure(1)
omega = logspace(-1,3,100);
sigma(I+L,'b-',W1/gam,'r--',T,'b-.',gam/W3,'r.',omega)
grid
legend('1/\sigma(S) performance', ...
'\sigma(W1) performance bound', ...
'\sigma(T) robustness', ...
'\sigma(1/W3) robustness bound')

% We show separately the singular values of the open-loop system L
% with respect to performance bound W1 and robustness bound 1/W3
% As a result of
% satisfying performance and robustness requirements, the smallest singular value of
% L lies above the bound W1 in the low frequency range and the largest singular value
% of L is below the bound 1/W3 in the high frequency range.
figure
omega = logspace(-1,3,100);
sigma(L,'b-',W1/gam,'r--',gam/W3,'r.',omega)
grid
legend('\sigma(L)','\sigma(W1) performance bound', ...
'\sigma(1/W3) robustness bound')

% Plot Step response of the uncertain Closed Loop
CL = looptransfer.CSo;
P_CL = looptransfer.PSi;

loopsens = series(K,G);
G_Cl = feedback(loopsens,eye(2));
figure
grid on;
hold on;
step(CL,'--r')
step(G_Cl,'--b')
% step of the system with no Control
% step(feedback(G,eye(2)))

% Plot Bode 
figure
step(feedback(G,eye(2)))
hold on;
grid on;
bode(G_Cl);
legend('Pre Control','Post Control')