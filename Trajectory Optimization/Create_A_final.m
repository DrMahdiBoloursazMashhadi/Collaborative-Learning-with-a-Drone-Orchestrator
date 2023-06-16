
function A_final= Create_A_final(x,k,coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,L,D,c1,eta1,H, alpha, T,wt,eta,M,variances)
    thershold = T/wt;
    ind3 = floor(T/thershold);
    for u=1:k
        e = eta1(u);
        sigma = (cc/(4*pi*fc))^2*e*(H^2+(coordinates(u,1)-x(2*ind3-1))^2+(coordinates(u,2)-x(2*ind3))^2)^(alpha/2);
        if u ==1
            z_jj =  D_k(u)*(1-exp(-(theta*B*N0)/(sigma*p_k)));
            z_kk =  D_k(u)*(exp(-(theta*B*N0)/(sigma*p_k)))*variances(u);
        else
            z_jj = D_k(u)*(1-exp(-(theta*B*N0)/(sigma*p_k))) +z_jj;
            z_kk = D_k(u)*(exp(-(theta*B*N0)/(sigma*p_k)))*variances(u) +z_kk;
        end

    end
    JJ = ((2*c1)/(L*D))*z_jj;
    KK = ((eta*M)/(D^2))*z_kk;
    A_final = JJ+KK;
end