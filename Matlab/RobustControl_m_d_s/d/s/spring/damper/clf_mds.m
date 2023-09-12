% Closed-loop frequency responses for three controllers of the
% mass/damper/spring system
%
sim_mds
omega = logspace(-2,2,100);
%
% H_infinity controller
clp_hin = lft(sim_ic,K_hin);
ref_hin = clp_hin(1,1).Nominal;
%
% Loop Shaping controller
clp_lsh = lft(sim_ic,K_lsh);
ref_lsh = clp_lsh(1,1).Nominal;
%
% mu-controller
clp_mu = lft(sim_ic,K_mu);
ref_mu = clp_mu(1,1).Nominal;
%
figure(1)
bode(ref_hin,'r-',ref_lsh,'m--',ref_mu,'b-.'), grid
title('Bode plots of closed-loop systems')
legend('H_\infty controller','Loop Shaping controller','\mu-controller',1)