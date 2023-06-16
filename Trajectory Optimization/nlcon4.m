function [c,ceq] = nlcon4(x, v_max, wt)
    
    itr1 = 1;index = 1;c=[];ceq=[];
   
    while itr1<wt
        c(index) = sqrt((x(itr1)-x(itr1+2))^2+((x(itr1+1)-x(itr1+3))^2))-v_max;
        index = index+1;
        itr1 = itr1+1;
    end
    
end