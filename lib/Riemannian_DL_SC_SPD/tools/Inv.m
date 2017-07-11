function P = Inv(Q)
    [U,E] = schur(Q); E = diag(E); E(abs(E)<1e-5) = 1e-5;
    P = U*diag(1./E)*U';
end