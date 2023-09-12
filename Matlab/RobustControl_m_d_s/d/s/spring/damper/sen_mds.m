% Frequency response of the sensitivity function
%
sim_mds
clp = lft(sim_ic,K);
%
% inverse performance weighting function
wts_mds
omega = logspace(-4,2,100);
%
% sensitivity function
sen_loop = clp(1,2);
sen64 = gridureal(sen_loop,'c',4,'k',4,'m',4);
figure(1)
bodemag(1/Wp,'r-',sen64,'b-',omega), grid
title('Closed-loop sensitivity function')
legend('Inverse weighting function','Sensitivity function',2)