function G = CalcG(point,k,coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,mu,L,c2,D,c1, eta1, H, alpha,eta,M,variances)
    %finding mean and variance of channel, denoting by z
    z_j = 0;z_k = 0;
    for itr=1:k
        e = eta1(itr);
        sigma = @(x) (cc/(4*pi*fc))^2*e*(H^2+(coordinates(itr,1)-x(1))^2+(coordinates(itr,2)-x(2))^2)^(alpha/2);
        if itr ==1
            z_j = @(x) D_k(itr)*(1-exp(-(theta*B*N0)/(sigma(x)*p_k)));
            z_k = @(x) D_k(itr)*(exp(-(theta*B*N0)/(sigma(x)*p_k)))*variances(itr);
        else
            z_j = @(x) D_k(itr)*(1-exp(-(theta*B*N0)/(sigma(x)*p_k))) +z_j(x);
            z_k = @(x) D_k(itr)*(exp(-(theta*B*N0)/(sigma(x)*p_k)))*variances(itr) +z_k(x);
        end
   
    end
    phi = @(x) 1-(mu/L)+((4*mu*c2)/(L*D))* z_j(x);
    objective = @(x) (((2*c1)/(L*D))*z_j(x)+(((eta*M)/(D^2))*z_k(x)))*(1/(1-phi(x))) ;
    G = objective(point);
end