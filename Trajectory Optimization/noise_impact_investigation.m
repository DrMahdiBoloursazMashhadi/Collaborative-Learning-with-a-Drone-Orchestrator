clear all
close all
clc

p_k=1e-4; % Users' maximum transmit power
N0=10^(-17.4-3); % Noise power spectrum density
B=2.5e6; % User Bandwidth
theta=10^(.053/10); % Waterfall threshold
mu=0.95;L=1;pi=3.14;fc=1e9;cc=3e8;
% c2=0.1;c1=1;
alpha = -3.4;
size_square = 70;
iteration_max=1;
problem = 0;H=20;d=10;ld = size_square/d;GapNormalized = [];PhiNormalized = [];fix_itrs = [1;2;3];n = 3;D = 5000;
v_max=25;run_max=1;WT=[5];T_max = 150;
eta = 0.1;M=28*28;
c2 = 0.5;
c1 = 1;
users = [5];Error_Path_total = [];
coordinates = [5 3;5 40;20 40;65 66;65 20];
D_k = [2000 300 300 1200 1200];
eta1 = [0.1 0.5 0.5 1 1];
%find weighted centroid
wcentroid_x = 0;wcentroid_y = 0;k=5;
for itr= 1:k
    wcentroid_x = (D_k(itr)/D)*coordinates(itr, 1) + wcentroid_x;
    wcentroid_y = (D_k(itr)/D)*coordinates(itr, 2) + wcentroid_y;

end
wc = [wcentroid_x wcentroid_y];start_point = wc;
for wt_itr=1:size(WT,2)
    wt = WT(wt_itr);
    x = sym('x',[1 2*wt]);
    for k_x=1:size(users,1)
        k = users(k_x);
        ATL_op = [];ATL_wc = [];ATL_op_comm = [];ind = 1;
        for psnr=5:2:30
            variances = [];
            
            variances(1) = (1) / (10 ^ (psnr / 10));
            variances(2) = (1) / (10 ^ (5 / 10));
            variances(3) = (1) / (10 ^ (5 / 10));
            variances(4) = (1) / (10 ^ (5 / 10));
            variances(5) = (1) / (10 ^ (5 / 10));
            
            All_Q_matrix = Create_Q_Matrix(x,k,coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,eta1,H, alpha, T_max,wt, mu,c2);
            All_A_matrix = Create_A_Matrix(x,k,coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,c1,eta1,H, alpha, T_max,wt,eta,M,variances);
            Q_prod = prod(All_Q_matrix);
            Q_prod = transpose(Q_prod);
            A_final = Create_A_final(x,k,coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,c1,eta1,H, alpha, T_max,wt,eta,M,variances);
            Objective_function = (All_A_matrix*Q_prod)+A_final;
 
            points_op = ErrorPathMinimizerWithFunc(x,Objective_function,start_point,size_square, v_max, wt);
            
            
            ATL_op(ind) = CalcErrorPathFunc(x,points_op,Objective_function);            
            ind=ind+1;
        end
    end
      
end
psnr = 5:2:30;
plot(psnr,ATL_op);

