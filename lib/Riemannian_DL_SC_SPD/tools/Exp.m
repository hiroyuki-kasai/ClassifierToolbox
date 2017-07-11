function Q = Exp(P)
    [U,E] = schur(P); E = diag(E);
    Q = U*diag(exp(E))*U';
end