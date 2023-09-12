%% LFT representation of a mass-damper-spring system with uncertainties

m_nom = 3; c_nom = 1; k_nom = 2;
p_m = 0.4; p_c = 0.2; p_k = 0.3;
mat_mi = [-p_m 1/m_nom; -p_m 1/m_nom];
mat_c = [0 c_nom; p_c c_nom];
mat_k = [0 k_nom; p_k k_nom];
int1 = nd2sys([1],[1 0]);
int2 = nd2sys([1],[1 0]);
systemnames = 'mat_mi mat_c mat_k int1 int2';
sysoutname = 'G';
inputvar = '[um;uc;uk;u]';
input_to_mat_mi = '[um;u-mat_c(2)-mat_k(2)]';
input_to_mat_c = '[uc;int1]';
input_to_mat_k = '[uk;int2]';
input_to_int1 = '[mat_mi(2)]';
input_to_int2 = '[int1]';
outputvar = '[mat_mi(1);mat_c(1);mat_k(1);int2]';
sysic