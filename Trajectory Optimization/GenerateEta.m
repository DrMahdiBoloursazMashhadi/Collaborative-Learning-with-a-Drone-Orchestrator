function etas_o = GenerateEta(k)
%     var = [];
%     temp =floor(k/n);
%     r = rem(k,n);
%     size = 0;
%     for itr1=1:temp
%         rng('shuffle')
%         rand = randperm(n,n);
%         for itr2=0:n-1
%             var(itr1*temp+itr2) = etas(rand(itr2+1));
%             size = size+1;
%         end
%     end
%     
%     rng('shuffle')
%     rand = randperm(n,r);
%     for itr3=1:r
%       var(size+1) = etas(rand(itr3));
%       size = size+1;
%     end
%     etas_o = var;
    var = [];
    for i=1:k
        if i==1
            rng('shuffle')
            var(i) = rand()*0.9+0.1;
        else
            var(i) = rand()*0.9+0.1;
            eq_r = 0;
            while(eq_r~=1)
                f = 0;
                for itr1=1:k
                    for itr2=itr1+1:size(var,1)
                        if var(itr1)==var(itr2)
                            rng('shuffle')
                            var(itr2) = rand()*0.9+0.1;
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
        
    end
    etas_o= var;
end