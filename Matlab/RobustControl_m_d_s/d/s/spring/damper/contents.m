%    FILES FOR ANALYSIS AND DESIGN OF MASS/DAMPER/SPRING SYSTEM
%               using Robust Control Toolbox, Version 3
%
%
% Building the open loop interconnection
%
% olp_mds.m  Creates model of the uncertain open loop system
%
% Controlers design
%
% hin_mds.m	 Design of Hinf controller
% lsh_mds.m	 Design of Hinf loop shaping controller
% ms_mds.m	 Design of mu-controller
%
% Controller order reduction
%
% red_mds.m	 Obtains controller of 4th order
%
% Analysis of the closed loop system
%
% rob_mds.m	 Robust stability analysis
% nrp_mds.m  Nominal performance and robust performance analysis
% wcp_mds.m	 Determination of the worst case performance
% frs_mds.m	 Frequency responses of the closed loop system
%            with nominal parameters
% clp_mds.m	 Transient responses of the closed loop system
%            with nominal parameters
%
% Characteristics of the uncertain system
%
% pfr_mds.m	 Frequency responses of the uncertain plant models
% pcf_mds.m  Bode plots of the uncertain closed loop system
% sen_mds.m  Sensitivity function of the closed loop system
% ppf_mds.m	 Singular values of the perturbed performance
% mcs_mds.m	 Transient responses for random values of uncertain
%            parameters
%
% Comparison of systems with three controllers
%
% kf_mds.m	 Frequency responses of the three controllers
% clf_mds.m	 Frequency responses of the three closed loop systems
% prf_mds.m	 Nominal performance of the three closed loop systems
% rbs_mds.m	 Robust stability of the three closed loop systems
% rbp_mds.m	 Robust performance of the three closed loop systems
%
% Auxiliary files
%
% mod_mds.m  Creates the uncertainty system model
% wts_mds.m	 Sets the performance weighting functions
% sim_mds.m	 Creates the simulation model of the closed loop system