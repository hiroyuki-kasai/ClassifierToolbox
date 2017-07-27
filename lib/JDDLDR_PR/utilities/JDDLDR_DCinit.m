
function [D,C]=JDDLDR_DCinit(X,numcomps,trls,lambda_a)
% the function of initialization

nClass      =  max(trls);
D           =  [];
C           =  [];
for ci      =  1:nClass
   cdat     =  X(:,trls==ci);
   [U,S,V]  =  svd(cdat);
   D(ci).M  =  U(:,1:numcomps);
   %D(ci).M  =  cdat;
   temD     =  D(ci).M;
   C(ci).M  =  inv(temD'*temD+lambda_a*eye(numcomps))*(temD'*cdat);
end