% Generate the open-loop connection for the
% mass/damper/spring system
%
mod_mds
%
% nominal frequency response of G
figure(1)
omega = logspace(-1,1,100);
bode(G.Nominal,'r-',G,'b--',omega), grid
title('Bode plot of G')
legend('Nominal system','Random samples')
%
% construct performance weighting function
wts_mds
omega = logspace(-4,4,100);
figure(2)
bodemag(1/Wp,'r-',omega), grid
title('Inverse of performance weighting function')
%
% open-loop connection with the weighting function
systemnames = ' G Wp Wu ';
inputvar = '[ dist; control ]';
outputvar = '[ Wp; -Wu; -G-dist ]';
input_to_G = '[ control ]';
input_to_Wp = '[ G+dist ]';
input_to_Wu = '[ control ]';
sys_ic = sysic;