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
v_max=25;run_max=1;WT=[1];T_max = 200;
eta = 0.1;M=28*28;
c2 = 0.1;
c1 = 1;
users = [5];PSNR = [30,5,5,5,5];variances = [];Error_Path_total = [];
coordinates = [5 3;5 40;20 40;65 66;65 20];
D_k = [2000 300 300 1200 1200];
eta1 = [0.1 0.5 0.5 1 1];
%find weighted centroid
wcentroid_x = 0;wcentroid_y = 0;k=5;
for var=1:k
    variances(var) = (1) / (10 ^ (PSNR(var) / 10));
end
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
        mkdir('C:\Users\mm03263\OneDrive - University of Surrey\Desktop\data\results\path-planning\',string(k))
        for r=1:run_max
            All_Q_matrix = Create_Q_Matrix(x,k,coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,eta1,H, alpha, T_max,wt, mu,c2);
            All_A_matrix = Create_A_Matrix(x,k,coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,c1,eta1,H, alpha, T_max,wt,eta,M,variances);
            Q_prod = prod(All_Q_matrix);
            Q_prod = transpose(Q_prod);
            A_final = Create_A_final(x,k,coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,c1,eta1,H, alpha, T_max,wt,eta,M,variances);
            Objective_function = (All_A_matrix*Q_prod)+A_final;
            
            
            points_op = ErrorPathMinimizerWithFunc(x,Objective_function,start_point,size_square, v_max, wt);
%                 if wt==1
%                     start_point = points_op;
%                 end
%             Error_Path_total(wt_itr)=CalcErrorPathFunc(x,points_op,Objective_function);
            
                                                                                                                      
            points_op_comm_t = GapCommminimizer(k,coordinates,wc,size_square,theta,B,N0,p_k,D_k,pi,fc,cc,mu,L,c2,D,c1,eta1,H, alpha);
            for wc_itr=1:T_max
                points_wc(2*wc_itr-1) = wc(1);
                points_wc(2*wc_itr) = wc(2);
                points_op_comm(2*wc_itr-1) = points_op_comm_t(1);
                points_op_comm(2*wc_itr) = points_op_comm_t(2);
            end
            index = 1;
            for wt1=1:wt
                for wt2=1:T_max/wt
                    points_op_f(index)=points_op(2*wt1-1);
                    points_wc_f(index)=points_wc(2*wt1-1);
                    points_op_comm_f(index)=points_op_comm(2*wt1-1);
                    index = index+1;
                    points_op_f(index)=points_op(2*wt1);
                    points_wc_f(index)=points_wc(2*wt1);
                    points_op_comm_f(index)=points_op_comm(2*wt1);
                    index = index+1;
                end
            end
            errors_op = [];errors_op_comm = [];errors_wc = [];errors_c = [];
            for round=1:T_max
                puav_op(1) = points_op_f(2*round-1); 
                puav_op(2) = points_op_f(2*round);
                puav_op_comm(1) = points_op_comm(2*round-1); 
                puav_op_comm(2) = points_op_comm(2*round);
                for u=1:k
                   
                    errors_op(round,u) = CalcError(theta, B, N0,p_k,cc,pi,fc,H,eta1(u),puav_op,[ coordinates(u,1) coordinates(u,2) ],alpha);
                    errors_op_comm(round,u) = CalcError(theta, B, N0,p_k,cc,pi,fc,H,eta1(u),puav_op_comm,[ coordinates(u,1) coordinates(u,2) ],alpha);
                    errors_wc(round,u) = CalcError(theta, B, N0,p_k,cc,pi,fc,H,eta1(u),wc,[ coordinates(u,1) coordinates(u,2) ],alpha);

                end
            end
                
            save(['C:\Users\mm03263\OneDrive - University of Surrey\Desktop\data\results\path-planning\200\' num2str(k) '/U_'  num2str(wt)  '.mat'],'points_op','puav_op_comm','wc','errors_op','errors_op_comm','errors_wc','D_k');

        end

    
    end
    
    
%     Error_Path_total(wt_itr,2)=CalcErrorPathFunc(x,points_wc_f,Objective_function);
%     Error_Path_total(wt_itr,3)=CalcErrorPathFunc(x,points_op_comm_f,Objective_function);
    
end
% for i=1:5
%     h(1)=plot(coordinates(i,1),coordinates(i,2),'red.');
%     hold on
% end
% for j=1:10
%     h(2)=plot(points_op(2*j-1),points_op(2*j),'green.');
%     text(points_op(2*j-1),points_op(2*j),string(j),'VerticalAlignment','bottom','HorizontalAlignment','center','color','black');
%     hold on
% end
% legend(h,{'users','drone'});