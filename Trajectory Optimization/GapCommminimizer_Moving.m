function gapmin= GapCommminimizer_Moving(x1,k,all_coordinates,centroid,size_square,theta,B,N0,p_k,D_k,pi,fc,cc,mu,L,c2,D,c1,eta1,H, alpha, wt, thershold)
   %finding mean and variance of channel, denoting by z
   z1 = 0;
   
   for counter=1:wt
        coordinates = cell2mat(all_coordinates(counter*thershold));
        for itr=1:k
            e = eta1(itr);
    %         mean = @(x) (4*pi*fc/cc)*(H^2+(coordinates(itr,1)-x(1))^2+(coordinates(itr,2)-x(2))^2)^-1;
            sigma = (cc/(4*pi*fc))^2*e*(H^2+(coordinates(itr,1)-x1(2*counter-1))^2+(coordinates(itr,2)-x1(2*counter))^2)^(alpha/2);
            z1 = log2(1+((sigma*p_k)/(B*N0))) +z1;
        end
   end
    objective = -z1 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    startPoint= zeros(1,2*wt);
    
    for itx=1:wt
        startPoint(2*itx-1) = centroid(1);
        startPoint(2*itx) = centroid(2);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    MyValue = 10e4;
    opts = optimoptions('fmincon','Display','iter','Algorithm','sqp', 'MaxFunctionEvaluations',MyValue);
    lb = zeros(1,2*wt);
    ub = size_square*ones(1,2*wt);
    g=matlabFunction(objective,'vars',{x1});
    nonlincon = [];
    gapmin = fmincon(g,startPoint,[],[],[],[],lb,ub,nonlincon,opts);
%     rng default % For reproducibility
%     gs = GlobalSearch;
%     lb = 0*ones(2);
%     ub = size_square*ones(2);
%     problem = createOptimProblem('fmincon','x0',centroid,...
%     'objective',objective,'lb',lb,'ub',ub);
%     gapmin = run(gs,problem);
%     AA = [];
%     bb = [];
%     Aeq = [];
%     beq = [];
%     lb = 0*ones(2);
%     ub = size_square*ones(2);
%     
%     options = optimset('Largescale','off','Display','iter');
%     nonlincon = @(x) nlcon(x, k, coordinates,mu,L,D,c2,D_k,theta,B,N0,p_k,fc,pi,cc,eta1,H,alpha, centroid);
%     x = fmincon(objective,centroid,AA,bb,Aeq,beq,lb,ub,nonlincon,options);
%     gapmin = x;
    
end