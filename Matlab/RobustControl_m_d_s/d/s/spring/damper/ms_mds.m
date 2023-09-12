% Mu-synthesis of the mass/damper/spring system
%
nmeas = 1;
ncont = 1;
mu_ic = sys_ic;
fv = logspace(-2,4,100);
opt = dkitopt('FrequencyVector',fv, ...
              'DisplayWhileAutoIter','on', ...
              'NumberOfAutoIterations',3)
[K_mu,CL_mu,bnd_mu,dkinfo] = dksyn(mu_ic,nmeas,ncont,opt);
K = K_mu;