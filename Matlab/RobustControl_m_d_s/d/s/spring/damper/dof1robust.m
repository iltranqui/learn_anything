% 1 dof system with 3 actuators 

m = 3;
c = 1;
k = 2;
pm = 0.4;
pc = 0.2;
pk = 0.3;
%
A=[ 0 1
-k/m -c/m];
B1 = [ 0 0 0
-pm -pc/m -pk/m];
B2 = [ 0
1/m];
C1 = [-k/m -c/m
0 c
k 0 ];
C2 = [ 1 0 ];
D11 = [-pm -pc/m -pk/m
0 0 0
0 0 0 ];
D12 = [1/m
0
0 ];
D21 = [0 0 0];
D22 = 0;


G = pck(A,[B1,B2],[C1;C2],[D11 D12;D21 D22])