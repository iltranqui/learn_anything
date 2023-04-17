% Performance weighting functions for the mass/damper/spring
% system
% The weighting functions Wp and Wu are used to reflect the relative
% significance of the performance requirement over different frequency ranges.
nuWp = [1  1.8  10  ];
dnWp = [1  8    0.01];
gainWp = 0.95;
Wp = gainWp*tf(nuWp,dnWp)
%
nuWu = 1;
dnWu = 1;
gainWu = 10^(-2);
Wu = gainWu*tf(nuWu,dnWu)