% Transient responses of the closed_loop
% mass/damper/spring system
%
sim_mds
clp_ic = lft(sim_ic,K);
%
% response to the reference
r = 1.0;    
ti = 0.01;    % time increment
tfin1 = 20.0; 
time1 = 0:ti:tfin1;
nstep1 = size(time1,2);
ref1(1:nstep1) = r;
tfin2 = 40.0; 
time2 = tfin1+ti:ti:tfin2;
nstep2 = size(time2,2);
ref2(1:nstep2) = 0.0;
tfin3 = 60.0;  % final time value
time3 = tfin2+ti:ti:tfin3;
nstep3 = size(time3,2);
ref3(1:nstep3) = r;
time = [time1,time2,time3];
ref = [ref1,ref2,ref3];
nstep = size(time,2);
dist(1:nstep) = 0.0;
[y,t] = lsim(clp_ic.Nominal,[ref',dist'],time);
%
figure(1)
plot(t,ref,'r--',t,y(:,1),'b-'), grid
title('Closed-loop transient response')
xlabel('Time (secs)')
ylabel('y (m)')
disp(['overshoot: ' num2str(100*(norm(y(:,1),inf)-1)) '%'])
clear ref1, clear ref2, clear ref3
clear dist
%
% response to the disturbance
ti = 0.01;    % time increment
tfin1 = 20.0; 
time1 = 0:ti:tfin1;
nstep1 = size(time1,2);
dist1(1:nstep1) = 1.0;
tfin2 = 40.0; 
time2 = tfin1+ti:ti:tfin2;
nstep2 = size(time2,2);
dist2(1:nstep2) = 0.0;
tfin3 = 60.0;  % final time value
time3 = tfin2+ti:ti:tfin3;
nstep3 = size(time3,2);
dist3(1:nstep3) = 1.0;
time = [time1,time2,time3];
dist = [dist1,dist2,dist3];
nstep = size(time,2);
ref(1:nstep) = 0.0;
[y,t] = lsim(clp_ic.Nominal,[ref',dist'],time);
figure(2)
plot(t,dist,'r--',t,y(:,1),'b-'), grid
title('Transient response to the disturbance')
xlabel('Time (secs)')
ylabel('y (m)')
clear dist1, clear dist2, clear dist3
clear ref