Petko Petkov (2023). Robust control design of the Mass/Damper/Spring system (https://www.mathworks.com/matlabcentral/fileexchange/10353-robust-control-design-of-the-mass-damper-spring-system), MATLAB Central File Exchange. Retrieved April 17, 2023.

This collection contains M-files intended for design of the Mass/Damper/Spring control system using the newly available functions from Robust Control Toolbox,version 3. Description of the system and version of the files using functions from mu-toolbox can be found in the book ?Robust Control Design with MATLAB? by Da-Wei Gu, Petko H. Petkov and Mihail M. Konstantinov, Springer-Verlag, London, 2005(http://www.springer.com/sgw/cda/frontpage/0,11855,4-192-22-46383093-0,00.html?changeHeader=true)

The book also presents other 5 case studies including robust control systems design of a triple inverted pendulum, a hard disk drive, a distillation column, a rocket system and a flexible-link manipulator. New codes of those designs exploiting Robust Control Toolbox, version 3.0 are available upon request.

These M-files should be used in the environment of MATLAB, version 7.1, Service Pack 3, together with the Robust Control Toolbox, version 3.0 and Control System Toolbox, version 6.2.

List of the files:

olp_mds.m Creates model of the uncertain open loop system
hin_mds.m Design of Hinf controller
lsh_mds.m Design of Hinf loop shaping controller
ms_mds.m Design of mu-controller
red_mds.m Obtains controller of 4th order
rob_mds.m Robust stability analysis
nrp_mds.m Nominal performance and robust performance analysis
wcp_mds.m Determination of the worst case performance
frs_mds.m Frequency responses of the closed loop system
with nominal parameters
clp_mds.m Transient responses of the closed loop system
with nominal parameters
pfr_mds.m Frequency responses of the uncertain plant models
pcf_mds.m Bode plot of the uncertain closed loop system
sen_mds.m Sensitivity function of the closed loop system
ppf_mds.m Singular values of the perturbed performance
mcs_mds.m Transient responses for random values of the uncertain
parameters
kf_mds.m Frequency responses of the three controllers
clf_mds.m Frequency responses of the three closed loop systems
prf_mds.m Nominal performance of the three closed loop systems
rbs_mds.m Robust stability of the three closed loop systems
rbp_mds.m Robust performance of the three closed loop systems
mod_mds.m Creates the uncertainty system model
wts_mds.m Sets the performance weighting functions
sim_mds.m Creates the simulation model of the closed loop system