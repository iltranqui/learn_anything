clear all;
close all;
clc;

% Building the Model

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

% Building the uncertain model
m_nom = 3; c_nom = 1; k_nom = 2;
p_m = 0.4; p_c = 0.2; p_k = 0.3;
mat_mi = [-p_m 1/m_nom; -p_m 1/m_nom];
mat_c = [0 c_nom; p_c c_nom];
mat_k = [0 k_nom; p_k k_nom];
int1 = nd2sys([1],[1 0]);
int2 = nd2sys([1],[1 0]);
systemnames = ’mat_mi mat_c mat_k int1 int2’;
sysoutname = ’G’;
inputvar = ’[um;uc;uk;u]’;
input_to_mat_mi = ’[um;u-mat_c(2)-mat_k(2)]’;
input_to_mat_c = ’[uc;int1]’;
input_to_mat_k = ’[uk;int2]’;
input_to_int1 = ’[mat_mi(2)]’;
input_to_int2 = ’[int1]’;
outputvar = ’[mat_mi(1);mat_c(1);mat_k(1);int2]’;
sysic;