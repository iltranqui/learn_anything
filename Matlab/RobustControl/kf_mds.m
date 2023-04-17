% Frequency responses of three controllers for the mass/damper/spring
% system
%
omega = logspace(-2,3,100);
%
figure(1)
bode(K_hin,'r-',K_lsh,'m--',K_mu,'c-.',omega), grid 
title('Bode plots of all controllers')
legend('H_\infty controller','Loop Shaping controller','\mu-controller',3)