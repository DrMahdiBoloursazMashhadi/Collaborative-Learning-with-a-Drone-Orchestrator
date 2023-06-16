function coordinates = GenerateCoordinates(k, ld,d)
%     RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*2)))
%     coefficients = randi([0,ld],2,k);
%     for itr=1:k
%         coordinates(itr,1) = d*coefficients(1,itr);
%         coordinates(itr,2) = d*coefficients(2,itr);
%     end
    rng('shuffle')
    coefficients_x = randi([0,ld],1,k);
    coefficients_y = randi([0,ld],1,k);
    for itr=1:k
        coordinates(itr,1) = d*coefficients_x(itr);
        coordinates(itr,2) = d*coefficients_y(itr);
    end
    eq_r = 0;
    while(eq_r~=1)
        f = 0;
        for itr1=1:k
            for itr2=itr1+1:k
                if coordinates(itr1,1)==coordinates(itr2,1) && coordinates(itr1,2)==coordinates(itr2,2)
                    rng('shuffle')
                    coefficients_x = randi([0,ld],1,1);
                    coefficients_y = randi([0,ld],1,1);
                    coordinates(itr2,1) = d*coefficients_x;
                    coordinates(itr2,2) = d*coefficients_y;
                    eq_r = 0;
                    f = 1;
                end
            end
        end
        if f==0
            eq_r = 1;
        end
    end
end