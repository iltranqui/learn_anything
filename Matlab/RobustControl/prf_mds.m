% Nominal performance for three controllers of the mass/damper/spring
% system
% 
clear all;
olp_mds;

omega = logspace(-2,2,100);
%
% H_infinity controller
clp_hin = lft(sys_ic,K_hin);
prf_hin = clp_hin(1,1).Nominal;
%
% Loop Shaping controller
clp_lsh = lft(sys_ic,K_lsh);
prf_lsh = clp_lsh(1,1).Nominal;
%
% mu-controller
clp_mu = lft(sys_ic,K_mu);
prf_mu = clp_mu(1,1).Nominal;
%
figure;
bodemag(prf_hin,'r-',prf_lsh,'m--', ...
        prf_mu,'b-.',omega), grid
title('Nominal performance: all controllers')
legend('H_\infty controller','Loop Shaping controller','\mu-controller',3)