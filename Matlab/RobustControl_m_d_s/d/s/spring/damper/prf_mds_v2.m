%
% Frequency responses of the perturbed plants
%

clear all; 

mod_mds
omega = logspace(-1,1,100);
[delta1,delta2,delta3] = ndgrid([-1 0 1],[-1 0 1], ...
[-1 0 1]);
for j = 1:27
delta = diag([delta1(j),delta2(j),delta3(j)]);
olp = starp(delta,G);
olp_ic = sel(olp,1,1);
olp_g = frsp(olp_ic,omega);
figure(1)
vplot('bode',olp_g,'c-')
subplot(2,1,1)
hold on
subplot(2,1,2)
hold ons
end
subplot(2,1,1)
olp_ic = sel(G,4,4);
olp_g = frsp(olp_ic,omega);
vplot('bode',olp_g,'r--')
subplot(2,1,1)
title('BODE PLOTS OF PERTURBED PLANTS')
hold off
subplot(2,1,2)
hold off