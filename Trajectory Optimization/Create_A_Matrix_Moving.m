
function All_A= Create_A_Matrix_Moving(x,k,all_coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,c1,eta1,H, alpha, T,wt,eta,M,variances)
    
    thershold = T/wt;
    itr1 = 1;
    while itr1<=T-1
        coordinates = cell2mat(all_coordinates(itr1));
        ind1 = floor(itr1/thershold);
        if mod( itr1 , thershold ) == 0
                ind1 = ind1-1;
        end
        for u=1:k
            e = eta1(u);
            sigma =  (cc/(4*pi*fc))^2*e*(H^2+(coordinates(u,1)-x(2*ind1+1))^2+(coordinates(u,2)-x(2*ind1+2))^2)^(alpha/2);
            if u ==1
                z_j =  D_k(u)*(1-exp(-(theta*B*N0)/(sigma*p_k)));
                z_k =  D_k(u)*(exp(-(theta*B*N0)/(sigma*p_k)))*variances(u);
            else
                z_j =  D_k(u)*(1-exp(-(theta*B*N0)/(sigma*p_k))) +z_j;
                z_k =  D_k(u)*(exp(-(theta*B*N0)/(sigma*p_k)))*variances(u) +z_k;
            end
            
        end
        J = ((2*c1)/(L*D))*z_j;
        K=((eta*M)/(D^2))*z_k;itr2 = itr1+1;
        All_A(1,itr1) =  (J+K);
        itr1 = itr1+1;
    end
   

end