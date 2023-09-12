% Builds the more complex open loop function to control the mass/damper/spring

wts_mds
mod_mds

systemnames = ' G Wp Wu ';
inputvar = '[ pert{3}; dist; control ]';
outputvar = '[ G(1:3); Wp; -Wu; -G(4)-dist ]';
input_to_G = '[ pert; control ]';
input_to_Wp = '[ G(4)+dist ]';
input_to_Wu = '[ control ]';
sysoutname = 'sys_ic'