function Q = Log(P)
    [U,E] = schur(P); E=diag(E); E(E<=0) = 1;
    Q = U*diag(log(E))*U';
end