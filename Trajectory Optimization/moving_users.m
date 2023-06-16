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
H=20;d=10;ld = size_square/d;D = 5000;
v_max=25;run_max=1;T_max = 150;

v_user_coefficient = 0.1;
eta = 0.8;M=28*28;
c2 = 0.5;
c1 = 1;
users = [5];noise_level = 10;PSNR = [noise_level,5,5,5,5];variances = [];
coordinates = [5 3;5 40;20 40;65 66;65 20];
% directions=[0,2,1,5,6];
directions=[0,0,1,2,0];
D_k = [2000 300 300 1200 1200];
eta1 = [0.1 0.5 0.5 1 1];
%find weighted centroid
k=5;
for var=1:k
    variances(var) = (1) / (10 ^ (PSNR(var) / 10));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
wcentroid_x = 0;wcentroid_y = 0;
for itr= 1:k
    wcentroid_x = (D_k(itr)/D)*coordinates(itr, 1) + wcentroid_x;
    wcentroid_y = (D_k(itr)/D)*coordinates(itr, 2) + wcentroid_y;
end
wc = [wcentroid_x wcentroid_y];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
start_point = wc;
% start_point =[28.3290   15.8772] ;%from wt=1 for wt=5 was used
% start_point =[28   15.8772] ;%from wt=1 for wt= 10,25 were used
% mkdir('C:\Users\mm03263\OneDrive - University of Surrey\Desktop\data\results\path-planning\moving\150\different_noise_levels\',string(noise_level))
V = GenerateUserSpeed(k,v_user_coefficient);
wt=10;x = sym('x',[1 2*wt]);All_coordinates={};x1 = sym('x1',[1 2*wt]);
for r=1:T_max
    coordinates = UpdateCoordinates(coordinates, k,directions,V);
    All_coordinates = [All_coordinates, coordinates];
end
for ii=1:150
    coord = cell2mat(All_coordinates(ii));
    for iii=1:5
        if coord(iii,1) < 0 || coord(iii,1)>size_square || coord(iii,2) < 0 || coord(iii,2)>size_square
            display('wrong bounds');
        end
    end
end
thershold = T_max/wt;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% for start point
All_Q_matrix = Create_Q_Matrix_Moving(x,k,All_coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,eta1,H, alpha, T_max,1, mu,c2);
All_A_matrix = Create_A_Matrix_Moving(x,k,All_coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,c1,eta1,H, alpha, T_max,1,eta,M,variances);
Q_prod = prod(All_Q_matrix);
Q_prod = transpose(Q_prod);
A_final = Create_A_final(x,k,cell2mat(All_coordinates(T_max)),theta,B,N0,p_k,D_k,pi,fc,cc,L,D,c1,eta1,H, alpha, T_max,1,eta,M,variances);
Objective_function = (All_A_matrix*Q_prod)+A_final;
start_point = ErrorPathMinimizerWithFunc(x,Objective_function,start_point,size_square, v_max, 1);
start_point(2) = floor(start_point(2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
All_Q_matrix = Create_Q_Matrix_Moving(x,k,All_coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,eta1,H, alpha, T_max,wt, mu,c2);
All_A_matrix = Create_A_Matrix_Moving(x,k,All_coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,c1,eta1,H, alpha, T_max,wt,eta,M,variances);
Q_prod = prod(All_Q_matrix);
Q_prod = transpose(Q_prod);
A_final = Create_A_final(x,k,cell2mat(All_coordinates(T_max)),theta,B,N0,p_k,D_k,pi,fc,cc,L,D,c1,eta1,H, alpha, T_max,wt,eta,M,variances);
Objective_function = (All_A_matrix*Q_prod)+A_final;
points_op = ErrorPathMinimizerWithFunc(x,Objective_function,start_point,size_square, v_max, wt);
points_op_comm = GapCommminimizer_Moving(x1,k,All_coordinates,wc,size_square,theta,B,N0,p_k,D_k,pi,fc,cc,mu,L,c2,D,c1,eta1,H, alpha, wt,thershold);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
points_op_a=[];index = 1;
for wt1=1:wt
    coord = cell2mat(All_coordinates(wt1*T_max/wt));
    for wt2=1:T_max/wt
        points_op_a(index,1)=points_op(2*wt1-1);       
        points_op_a(index,2)=points_op(2*wt1);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        wcentroid_x = 0;wcentroid_y = 0;
        for itr= 1:k
            wcentroid_x = (D_k(itr)/D)*coord(itr, 1) + wcentroid_x;
            wcentroid_y = (D_k(itr)/D)*coord(itr, 2) + wcentroid_y;
        end
        points_wc_a(index,1) = wcentroid_x;
        points_wc_a(index,2) = wcentroid_y;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        points_op_comm_a(index,1)=points_op_comm(2*wt1-1); 
        points_op_comm_a(index,2)=points_op_comm(2*wt1);
        index = index+1;
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
errors_op = [];errors_op_comm = [];errors_wc = [];errors_c = [];

for round=1:T_max
    coordinates = cell2mat(All_coordinates(round));
    for u=1:k
        errors_op(round,u) = CalcError(theta, B, N0,p_k,cc,pi,fc,H,eta1(u),points_op_a(round,:),[ coordinates(u,1) coordinates(u,2) ],alpha);
        errors_op_comm(round,u) = CalcError(theta, B, N0,p_k,cc,pi,fc,H,eta1(u),points_op_comm_a(round,:),[ coordinates(u,1) coordinates(u,2) ],alpha);
        errors_wc(round,u) = CalcError(theta, B, N0,p_k,cc,pi,fc,H,eta1(u),points_wc_a(round,:),[ coordinates(u,1) coordinates(u,2) ],alpha);

    end
end
% save(['C:\Users\mm03263\OneDrive - University of Surrey\Desktop\data\results\path-planning\moving\150\new plot results\' num2str(k) '/U_'  num2str(wt)  '.mat'],'errors_op','errors_op_comm','errors_wc','All_coordinates','points_op_a','points_wc_a','points_op_comm_a','V');
save(['C:\Users\mm03263\OneDrive - University of Surrey\Desktop\data\results\path-planning\moving\150\different_noise_levels\' num2str(noise_level)  '.mat'],'errors_op','errors_op_comm','errors_wc','All_coordinates','points_op_a','points_wc_a','points_op_comm_a','V');
% ind = 10*15;
% coord = cell2mat(All_coordinates(ind));
% for u=1:5
%     h(1) = plot(coord(u,1),coord(u,2),'cD','MarkerSize',5);
%     text(coord(u,1),coord(u,2),string(D_k(u)),'VerticalAlignment','bottom','HorizontalAlignment','center','color','black');
%     hold on
% end
% h(2)=plot(points_op_a(ind,1),points_op_a(ind,2),'redD','MarkerSize',5);
% hold on
% h(3)=plot(points_wc_a(ind,1),points_wc_a(ind,2),'blueD','MarkerSize',5);
% hold on
% h(4)=plot(points_op_comm_a(ind,1),points_op_comm_a(ind,2),'blackD','MarkerSize',5);
% legend(h,{'Users','Optimized solution','Data weighted centroid', 'Maximum rate'});