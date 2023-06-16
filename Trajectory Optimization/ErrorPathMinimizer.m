function gapmin= ErrorPathMinimizer(k,coordinates,centroid,size_square,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,c1,eta1,H, alpha, v_max, T, mu,c2,eta,M,variances)
    total = 0;itr1 = 1;
    while itr1<2*T-1
        
        for u=1:k
            e = eta1(u);
            sigma = @(x) (cc/(4*pi*fc))^2*e*(H^2+(coordinates(u,1)-x(itr1))^2+(coordinates(u,2)-x(itr1+1))^2)^(alpha/2);
            if u ==1
                z_j = @(x) D_k(u)*(1-exp(-(theta*B*N0)/(sigma(x)*p_k)));
                z_k = @(x) D_k(u)*(exp(-(theta*B*N0)/(sigma(x)*p_k)))*variances(u);
            else
                z_j = @(x) D_k(u)*(1-exp(-(theta*B*N0)/(sigma(x)*p_k))) +z_j(x);
                z_k = @(x) D_k(u)*(exp(-(theta*B*N0)/(sigma(x)*p_k)))*variances(u) +z_k(x);
            end
            
        end
        J = @(x)((2*c1)/(L*D))*z_j(x);
        K=@(x)((eta*M)/(D^2))*z_k(x);itr2 = itr1+2;
        while itr2<2*T+1
            for u=1:k
                e = eta1(u);
                sigma = @(x) (cc/(4*pi*fc))^2*e*(H^2+(coordinates(u,1)-x(itr2))^2+(coordinates(u,2)-x(itr2+1))^2)^(alpha/2);
                if u ==1
                    z2 = @(x) D_k(u)*(1-exp(-(theta*B*N0)/(sigma(x)*p_k))); 
                else
                    z2 = @(x) D_k(u)*(1-exp(-(theta*B*N0)/(sigma(x)*p_k))) +z2(x);
                end
            end
            phi = @(x) 1-(mu/L)+((4*mu*c2)/(L*D))* z2(x);
            if itr2==3
                multi = @(x) phi(x);
            else
                multi = @(x) phi(x)*multi(x);
            end
            itr2 = itr2+2;
        end
        if itr1==1
            total = @(x) (J(x)+K(x))*multi(x);
        else
            total = @(x) (J(x)+K(x))*multi(x)+total(x);
        end
        itr1 = itr1+2;
    end
    %%%%%%%%%%%%%%%%%%%%%%
   
    for u=1:k
        e = eta1(u);
        sigma = @(x) (cc/(4*pi*fc))^2*e*(H^2+(coordinates(u,1)-x(2*T-1))^2+(coordinates(u,2)-x(2*T))^2)^(alpha/2);
        if u ==1
            z_jj = @(x) D_k(u)*(1-exp(-(theta*B*N0)/(sigma(x)*p_k)));
            z_kk = @(x) D_k(u)*(exp(-(theta*B*N0)/(sigma(x)*p_k)))*variances(u);
        else
            z_jj = @(x) D_k(u)*(1-exp(-(theta*B*N0)/(sigma(x)*p_k))) +z_jj(x);
            z_kk = @(x) D_k(u)*(exp(-(theta*B*N0)/(sigma(x)*p_k)))*variances(u) +z_kk(x);
        end

    end
    JJ = @(x)((2*c1)/(L*D))*z_jj(x);
    KK = @(x)((eta*M)/(D^2))*z_kk(x);
    %%%%%%%%%%%%%%%%%%%%%%
    objective = @(x) total(x)+JJ(x)+KK(x) ;
   
    ms = MultiStart;
    opts = optimoptions(@fmincon,'Algorithm','interior-point');
    lb = zeros(1,2*T);
    ub = size_square*ones(1,2*T);
    startPoint= zeros(1,2*T);
    startPoint(1) = centroid(1);
    startPoint(2) = centroid(2);
    nonlincon = @(x) nlcon4(x, v_max, T);
%     problem = createOptimProblem('fmincon','x0',startPoint,...
%     'objective',objective,'lb',lb,'ub',ub,...
%     'options',opts,'nonlcon',nonlincon);
%     rng default % for reproducibility
%     [x,f] = run(ms,problem,1000); 
%     gapmin = x;
    gapmin = fmincon(objective,startPoint,[],[],[],[],lb,ub,nonlincon,opts);

end