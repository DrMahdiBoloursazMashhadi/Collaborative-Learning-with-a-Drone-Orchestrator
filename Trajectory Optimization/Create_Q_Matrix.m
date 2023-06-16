function all_Q= Create_Q_Matrix(x,k,coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,eta1,H, alpha, T,wt, mu,c2)
   
   thershold = T/wt;
   itr2=2; all_Q=sym(ones(T-1,T-1));
  
   while itr2<=T
        ind2 = floor(itr2/thershold);
        if mod( itr2 , thershold ) == 0
            ind2 = ind2-1;
        end
        for u=1:k
            e = eta1(u);
            sigma =  (cc/(4*pi*fc))^2*e*(H^2+(coordinates(u,1)-x(2*ind2+1))^2+(coordinates(u,2)-x(2*ind2+2))^2)^(alpha/2);
            if u ==1
                z2 =  D_k(u)*(1-exp(-(theta*B*N0)/(sigma*p_k))); 
            else
                z2 =  D_k(u)*(1-exp(-(theta*B*N0)/(sigma*p_k))) +z2;
            end
        end
        phi =  1-(mu/L)+((4*mu*c2)/(L*D))* z2;
        for row=1:itr2-1
          all_Q(itr2-1,row) = phi;
          
        end
        itr2 = itr2+1;
            
    end
   
   
end

