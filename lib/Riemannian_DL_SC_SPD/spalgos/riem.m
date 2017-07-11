function [d t] = riem(A,B)

t=tic;

% the eig without 'chol' is the best so far!
d = eig(A, B); 
d=norm(log(d));

%v = eig(A*inv(B));




% t=tic;
% r = chol(B);
% ir = inv(r);
% ra = chol(A);
%[u s v]=svd(ir*ra);
%u=r'*v*diag(log(diag(s)))*v'*ir;
% [u d]=eig(ir'*ra'*ra*ir);
% m=r'*u*diag(log(diag(d)))*u'*ir';
%d=norm(u(:));

t=toc(t);