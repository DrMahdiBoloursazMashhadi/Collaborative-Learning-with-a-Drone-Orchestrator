unction gapmin= ErrorPathMinimizerWithFunc(x,objective,centroid,size_square, v_max, wt)
    
    
    ms = MultiStart;
    MyValue = 10e4;
    opts = optimoptions('fmincon','Display','iter','Algorithm','sqp', 'MaxFunctionEvaluations',MyValue);
    lb = zeros(1,2*wt);
    ub = size_square*ones(1,2*wt);
    startPoint= zeros(1,2*wt);
    
    for itx=1:wt
        startPoint(2*itx-1) = centroid(1);
        startPoint(2*itx) = centroid(2);
    end
    nonlincon = @(x) nlcon4(x, v_max, wt);
    g=matlabFunction(objective,'vars',{x});
%     problem = createOptimProblem('fmincon','x0',startPoint,...
%     'objective',g,'lb',lb,'ub',ub,...
%     'options',opts,'nonlcon',nonlincon);
%     rng default % for reproducibility
%     [x,f] = run(ms,problem,1);
%     gapmin = x;
    rng default % For reproducibility
    gs = GlobalSearch;
    startPointIntl = fmincon(g,startPoint,[],[],[],[],lb,ub,nonlincon,opts);
    problem = createOptimProblem('fmincon','x0',startPointIntl,...
    'objective',g,'lb',lb,'ub',ub,'options',opts,'nonlcon',nonlincon);
    gapmin = run(gs,problem);
%     gapmin = fmincon(g,startPoint,[],[],[],[],lb,ub,nonlincon,opts);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
end