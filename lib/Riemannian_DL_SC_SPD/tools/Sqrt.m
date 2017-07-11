function Q = Sqrt(P)
    [U,E] = schur(P); E=diag(E); E(E<=0) = 0;
    Q = U*diag(sqrt(E))*U';
end