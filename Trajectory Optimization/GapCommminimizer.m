function gapmin= GapCommminimizer(k,coordinates,centroid,size_square,theta,B,N0,p_k,D_k,pi,fc,cc,mu,L,c2,D,c1,eta1,H, alpha)
   %finding mean and variance of channel, denoting by z
   z1 = 0;
    for itr=1:k
        e = eta1(itr);
%         mean = @(x) (4*pi*fc/cc)*(H^2+(coordinates(itr,1)-x(1))^2+(coordinates(itr,2)-x(2))^2)^-1;
        sigma = @(x) (cc/(4*pi*fc))^2*e*(H^2+(coordinates(itr,1)-x(1))^2+(coordinates(itr,2)-x(2))^2)^(alpha/2);
        if itr ==1
            z1 = @(x) log2(1+((sigma(x)*p_k)/(B*N0))); 
        else
            z1 = @(x) log2(1+((sigma(x)*p_k)/(B*N0))) +z1(x);
        end
   
    end
    objective = @(x) -z1(x) ;
    ms = MultiStart;
    opts = optimoptions(@fmincon,'Algorithm','interior-point');
    lb = [0 0];
    ub = [size_square size_square];
    nonlincon = [];
    gapmin = fmincon(objective,centroid,[],[],[],[],lb,ub,nonlincon,opts);
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