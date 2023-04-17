% Controller order reduction for the mass/damper/spring system
%
omega = logspace(-2,4,100);
[Kred,redinfo] = reduce(K,4);
%
figure(1)
bode(K,'r-',Kred,'c--'), grid
title('Bode plots of full-order and reduced-order controllers')
legend('Full order controller','Reduced order controller',2)
[gap,nugap] = gapmetric(K,Kred) 
K = Kred;