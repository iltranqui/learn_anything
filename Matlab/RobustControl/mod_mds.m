% Uncertain model of the Mass/Damper/Spring system
%
m = ureal('m',3,'Percentage',40);
c = ureal('c',1,'Percentage',20);
k = ureal('k',2,'Percentage',30);
%
u = icsignal(1);
x = icsignal(1);
xdot = icsignal(1);
M = iconnect;
M.Input = u;
M.Output = x;
M.Equation{1} = equate(x,tf(1,[1,0])*xdot);
M.Equation{2} = equate(xdot,tf(1/m,[1,0])*(u-k*x-c*xdot));
G = M.System;