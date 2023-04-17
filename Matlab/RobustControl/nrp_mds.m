% Nominal and robust performance of the
% mass/damper/spring system
%
clp_ic = lft(sys_ic,K);;
omega = logspace(-1,2,100);
clp_g = ufrd(clp_ic,omega);
%
% nominal performance
figure(1)
sv = sigma(clp_ic.Nominal,omega);
sys_frd = frd(sv(1,:),omega);
semilogx(sys_frd,'r-')
grid
title('Nominal performance')
xlabel('Frequency (rad/s)')
%
% robust performance
opt = robopt('Display','on');
[perfmarg,perfmargunc,report,info] = robustperf(clp_g,opt);
report
figure(2)
semilogx(info.MussvBnds(1,1),'r-',info.MussvBnds(1,2),'b--')
grid
title('Robust performance')
xlabel('Frequency (rad/s)')
ylabel('mu')
legend('\mu-upper bound','\mu-lower bound',3)