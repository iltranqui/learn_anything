% H_infinity design for the mass/damper/spring system
%
nmeas = 1;
ncon = 1;
gmin = 0.1;
gmax = 10;
tol = 0.001;
%hin_ic = sys_ic.Nominal;
hin_ic=sel(sys_ic,[4:6],[4:5])
[K_hin,clp] = hinfsyn(hin_ic,nmeas,ncon,gmin,gmax,tol);
disp(' ')
get(K_hin)
disp(' ')
disp('Closed-loop poles')
sp = pole(clp)
omega = logspace(-2,6,100);
sigma(clp,'m-',omega), grid
title('Singular Value Plot of clp')
K = K_hin;