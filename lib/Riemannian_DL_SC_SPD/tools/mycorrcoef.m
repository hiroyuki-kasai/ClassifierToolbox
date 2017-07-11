function rho = mycorrcoef(x1,x2)
R = corrcoef(x1,x2);
rho = R(1,2);
end