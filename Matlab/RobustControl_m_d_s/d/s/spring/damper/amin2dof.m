% H inf syntheiss iwht 2dof, so with 2 degree controllers

systemnames = ' G M Wp Wu ';
inputvar = '[ ref; dist; control ]';
outputvar = '[ Wp; Wu; ref; G+dist ]';
input_to_G = '[ control ]';
input_to_M = '[ ref ]';
input_to_Wp = '[ G+dist-M ]';
input_to_Wu = '[ control ]';
sys_ic = sysic;

nmeas = 2;
ncont = 1;
gmin = 0.1;
gmax = 10;
tol = 0.001;
[K,clp] = hinfsyn(sys_ic.Nominal,nmeas,ncont,gmin,gmax);
