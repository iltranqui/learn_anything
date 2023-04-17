% Performance of the uncertain closed-loop mass/damper/spring systems
%
% uncertain system peformance
omega = logspace(-2,2,100);
clp_ic = lft(sys_ic,K);
clp64 = gridureal(clp_ic,'c',4,'k',4,'m',4);
%
% robust performance
rp = ucomplexm('rp',zeros(1,2));
clp2 = lft(clp_ic,rp);
[M,Delta,blkstruct] = lftdata(clp2);
Mfrd = frd(M,omega);
bnd_mu = mussv(Mfrd(1:5,1:4),blkstruct);
%
figure(1)
sigma(clp64,'b--',bnd_mu(:,1),'r-',omega), grid 
title('Uncertain system performance')
legend('Uncertain system performance','Robust performance',3)