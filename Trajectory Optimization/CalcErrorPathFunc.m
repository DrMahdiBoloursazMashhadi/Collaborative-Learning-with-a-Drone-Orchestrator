function error= CalcErrorPathFunc(x,points,objective)
    g=matlabFunction(objective,'vars',{x});
    error = g(points);
end