function L=sqrtpsdinv(X)

   [u,d]=schur(X);
   d=diag(d);
   d(d<=0)=inf;
   L=u*diag(1./sqrt(d))*u';
end
