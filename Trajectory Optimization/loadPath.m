clear all
close all
clc

p_k=1e-4; % Users' maximum transmit power
N0=10^(-17.4-3); % Noise power spectrum density
B=1.5e6; % User Bandwidth
theta=10^(.023/10); % Waterfall threshold
mu=0.95;L=1;pi=3.14;fc=1e9;cc=3e8;
% c2=0.1;c1=1;
alpha = -3.4;
size_square = 70;
iteration_max=1000;
problem = 0;H=20;d=10;ld = size_square/d;GapNormalized = [];PhiNormalized = [];fix_itrs = [1;2;3];n = 3;D = 60000;
v_max=10;run_max=1;T_max=5;
users = [5];
Gap_total_v_op = zeros(1,2*v_max+1);
Gap_total_v_wc = zeros(1,2*v_max+1);

for k_x=1:size(users,1)
    k = users(k_x);
    
    for r=1:run_max
        filename = strcat('C:\Users\mm03263\OneDrive - University of Surrey\Desktop\results\PathPlanning\', num2str(k), '/U_' , num2str(k) , 'run_' ,num2str(r) , '.mat');
        myVars = {'Gap_v_wc','Gap_v_op'};
        S = load(filename,myVars{:});
        Gap_total_v_op = (S.Gap_v_op) + Gap_total_v_op;
       Gap_total_v_wc = (S.Gap_v_wc) +Gap_total_v_wc;
    end
    for itr = 1:2*v_max+1
        Gap_total_v_op(itr) = Gap_total_v_op(itr)/run_max;
        Gap_total_v_wc(itr) = Gap_total_v_wc(itr)/run_max;
    end
    
    
end




% x = categorical({'3','5','7','11','15'});
% x = reordercats(x,{'3','5','7','11','15'});
% 
% h1 = bar(x,GapNormalized);
% xlabel('No. Users') 
% ylabel('Normalized Gap')
% ylim([0.5 1]);

v=0:0.5:v_max;
plot(v,Gap_total_v_wc) 
hold on 
plot(v,Gap_total_v_op)
legend('Weighted centroid','Optimized location')
xlabel('Max speed') 
ylabel('Path error')