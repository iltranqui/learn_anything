% Robust stability of the closed-loop mass/damper/spring system
%
omega = logspace(-1,2,100);
%
% Hinf controller
clp_hin = lft(sys_ic,K_hin);
[M,Delta,blkstruct] = lftdata(clp_hin);
Mfrd = frd(M,omega);
rbnds_hin = mussv(Mfrd(1:3,1:3),blkstruct);
%
% Loop Shaping controller
clp_lsh = lft(sys_ic,K_lsh);
[M,Delta,blkstruct] = lftdata(clp_lsh);
Mfrd = frd(M,omega);
rbnds_lsh = mussv(Mfrd(1:3,1:3),blkstruct);
%
% mu-controller
clp_mu = lft(sys_ic,K_mu);
[M,Delta,blkstruct] = lftdata(clp_mu);
Mfrd = frd(M,omega);
rbnds_mu = mussv(Mfrd(1:3,1:3),blkstruct);
%
figure(1)
loglog(rbnds_hin(:,1),'r-',rbnds_lsh(:,1),'m--',rbnds_mu(:,1),'b-.')
grid
xlabel('Frequency (rad/s)')
ylabel('Upper bound of \mu')
title('Robust stability for all controllers')
legend('H_\infty controller','Loop Shaping controller','\mu-controller',4)