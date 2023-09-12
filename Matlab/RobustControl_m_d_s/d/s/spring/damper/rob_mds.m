% Robust stability analysis of the mass/damper/spring
% system
%
clp_ic = lft(sys_ic,K);
omega = logspace(-1,2,100);
clp_g = ufrd(clp_ic,omega);
%
opt = robopt('Display','on');
[stabmarg,destabu,report,info] = robuststab(clp_g,opt);
report
figure(1)
loglog(info.MussvBnds(1,1),'r-',info.MussvBnds(1,2),'b--')
grid
title('Robust stability')
xlabel('Frequency (rad/s)')
ylabel('mu')
legend('\mu-upper bound','\mu-lower bound',3)