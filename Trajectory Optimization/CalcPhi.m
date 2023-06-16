function Phi = CalcPhi(point, k,coordinates,theta,B,N0,p_k,D_k,pi,fc,cc,mu,L,c2,D,eta1, H, alpha)
    %finding mean and variance of channel, denoting by z
    z1 = 0;
    for itr=1:k
        e = eta1(itr);
%         mean = @(x) (4*pi*fc/cc)*(H^2+(coordinates(itr,1)-x(1))^2+(coordinates(itr,2)-x(2))^2)^-1;
        sigma = @(x) (cc/(4*pi*fc))^2*e*(H^2+(coordinates(itr,1)-x(1))^2+(coordinates(itr,2)-x(2))^2)^(alpha/2);
        if itr ==1
            z1 = @(x) c2(itr)*D_k(itr)*(1-exp(-(theta*B*N0)/(sigma(x)*p_k))); 
        else
            z1 = @(x) c2(itr)*D_k(itr)*(1-exp(-(theta*B*N0)/(sigma(x)*p_k))) +z1(x);
        end
   
    end
    objective = @(x) 1-(mu/L)+((4*mu)/(L*D))* z1(x);
  
    Phi = objective(point);
    if objective(point) > 1
        
        violation = 1
    end
    
end