% Generate the open-loop connection for the mass/damper/spring
% system simulation
%
mod_mds
systemnames = ' G ';
inputvar = '[ ref; dist; control ]';
outputvar = '[ G+dist; ref-G-dist ]';
input_to_G = '[ control ]';
sim_ic = sysic;