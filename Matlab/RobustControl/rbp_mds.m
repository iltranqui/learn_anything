% Robust performance comparison for three
% controllers of the mass/damper/spring system
%
omega = logspace(-2,2,100);
rp = ucomplexm('rp',zeros(1,2));
%
% H_infinity controller
clp_hin = lft(sys_ic,K_hin,1,1);
clp2 = lft(clp_hin,rp);
[M,Delta,blkstruct] = lftdata(clp2);
Mfrd = frd(M,omega);
bnd_hin = mussv(Mfrd(1:5,1:4),blkstruct);
%
% Loop Shaping controller
clp_lsh = lft(sys_ic,K_lsh,1,1);
clp2 = lft(clp_lsh,rp);
[M,Delta,blkstruct] = lftdata(clp2);
Mfrd = frd(M,omega);
bnd_lsh = mussv(Mfrd(1:5,1:4),blkstruct);
%
% mu-controller
clp_mu = lft(sys_ic,K_mu,1,1);
clp2 = lft(clp_mu,rp);
[M,Delta,blkstruct] = lftdata(clp2);
Mfrd = frd(M,omega);
bnd_mu = mussv(Mfrd(1:5,1:4),blkstruct);
%
figure(1)
semilogx(bnd_hin(:,1),'r-',bnd_lsh(:,1),'m--',bnd_mu(:,1),'b-.')
grid
xlabel('Frequency (rad/s)')
ylabel('Upper bound of \mu')
title('Robust performance for three controllers')
legend('H_\infty controller','Loop Shaping controller','\mu-controller',1)