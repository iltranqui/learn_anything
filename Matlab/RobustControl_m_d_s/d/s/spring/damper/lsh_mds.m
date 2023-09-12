% Loop Shaping Design for the mass/damper/spring system
%
mod_mds
%
% set the pre-compensator
nuW1 = [2   1];
dnW1 = [0.9 0];
gainW1 = 8;
W1 = gainW1*tf(nuW1,dnW1);
%
% frequency response of W1
omega = logspace(-2,4,100);
figure(1)
bodemag(W1,'r-',omega), grid
title('Frequency response of the precompensator')
%
% compute the loop shaping controller
[K_0,cl,gam,info] = ncfsyn(G.Nom,W1);
emax = info.emax;
disp(['The nugap robustness emax = ' num2str(emax)]);
%
% frequency responses of the plant and shaped plant
Gs = info.Gs;
omega = logspace(-1,2,200);
figure(2)
bodemag(G.Nom,'b-',Gs,'r--',omega), grid
title('Frequency responses of the plant and shaped plant')
legend('Original plant','Shaped plant')
%
% obtain the negative feedback controller
K_lsh = -K_0; K = K_lsh;