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
% How to erite an uncertain system
systemnames = ' G Wp Wu ';              % the names of the TF to insert
inputvar = '[ dist; control ]';         % the names of the input TFs
outputvar = '[ Wp; -Wu; -G-dist ]';     % the names of the output TFs
input_to_G = '[ control ]';             % Since G has only 1 input 
input_to_Wp = '[ G+dist ]';             % only 1 input
input_to_Wu = '[ control ]';            % only 1 input
sys_ic = sysic;

