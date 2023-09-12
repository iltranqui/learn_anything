% Frequency responses of the perturbed mass/damper/spring
% open-loop systems
%

% pfr_mds.m Frequency responses of the uncertain plant models
mod_mds
omega = logspace(-1,1,100);
G64 = gridureal(G,'c',4,'k',4,'m',4);
%
figure(1)
bode(G.Nominal,'r-',G64,'b--',omega), grid
title('Bode plots of uncertain plant')
legend('Nominal plant','Uncertain plant')