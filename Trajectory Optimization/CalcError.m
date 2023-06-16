function error = CalcError(theta, B, N0,p_k,cc,pi,fc,H,eta1,puav,coordinate,alpha)
    sigma =  (cc/(4*pi*fc))^2*(eta1)*(H^2+(coordinate(1)-puav(1))^2+(coordinate(2)-puav(2))^2)^(alpha/2);
    error = (1-exp(-(theta*B*N0)/(sigma*p_k)));
end