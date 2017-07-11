function L=logpsd(X)

   [u,d]=schur(X);
   d=diag(d);
   d(d<=0)=1;
   L=u*diag(log(d))*u';
end
