function V = GenerateUserSpeed(k,v_user_coefficient)
    for itr=1:k
%         rng('shuffle')
        v_x = v_user_coefficient*rand(1,1);
        v_y = v_user_coefficient*rand(1,1);
        V(itr,1) = v_x ;
        V(itr,2) = v_y ;
    end
end