% Frequency responses of the perturbed closed-loop
% mass/damper/spring systems
%
sim_mds
omega = logspace(-1,2,100);
%
clp_ic = lft(sim_ic,K);
clp64 = gridureal(clp_ic,'c',4,'k',4,'m',4);
%
figure(1)
bode(clp_ic(1,1).Nominal,'r-',clp64(1,1),'b--',omega), grid
title('Bode plots of the closed loop system')
legend('Nominal system','Uncertain system',3)