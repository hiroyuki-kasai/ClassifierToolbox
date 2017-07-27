function p = JDDLDR_UP3(X,pi,ZA,ZB,D,C,gamma1,gamma2,MaxIteration_num)
% X: original training data

beta = 0.001;
A  = [];
for ci = 1:size(C,2)
    A = [A D(ci).M*C(ci).M];
end
  
S = gamma1*ZA*ZA'+gamma2*ZB*ZB';
% S = gamma1*ZA*ZA';
p = pi;
numcomps = size(p,2);

iter_num_sub= 1;
phi1 = X*X';
phi2 = 2*X*A';
  
while iter_num_sub<MaxIteration_num % subject iteration for updating p, iteration ends when the contiguous function get close enough or reach teh max_iteration
    phi = (X-p*A)*(X-p*A)';
%     phi = eye(size(phi));
    [U,V] = eig(phi-S); p1 = U(:,1:numcomps);
      sum(sum(abs(p+beta*(p1-p) - p)));
      p = p+beta*(p1-p);
%     p = p1;
%     clear U V W phi
    iter_num_sub=iter_num_sub+1;
end
