% Frequency responses of the closed-loop mass/damper/spring system
%
sim_mds
clp = lft(sim_ic,K);
%
% Bode plots of the closed loop systems
ref_loop = clp(1,1);
omega = logspace(-2,4,100);
figure(1)
bode(ref_loop.Nominal,omega), grid
title('Bode plots of the closed loop system')
%
% sensitivity function
sen_loop = clp(1,2);
omega = logspace(-4,2,100);
figure(2)
bodemag(sen_loop.Nominal,omega), grid
title('Sensitivity function frequency response')
%
% controller frequency response
figure(3)
omega = logspace(-4,2,100);
bode(K,omega), grid
title('Controller frequency response')