%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function obj = objfun_ff(alpha,y,train_decomp,lambda,initial_alpha,method,theta)
% this function compute the objective function with given parameters
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input parameters:
% alpha: the optimized adjustment parameters
% y: the training label
% train_decomp: the eigen decomposition of the training data
% lambda:  the regularizer
% initial_alpha: the initial value of alpha to regularize alpha
% method: which criterion to be used
% theta: the kernel parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output parameters:
% obj: the objective value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jianjia Zhang, jz163@uowmail.edu.au Dec, 2014, all rights reserved
% For implementation details, please refer to: 
% "Learning Discriminative Stein Kernel for SPD Matrices and Its Applications." 
% arXiv preprint arXiv:1407.1974 (2014).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function obj = objfun_ff_new(alpha,y,train_decomp,lambda,initial_alpha,method,theta)
[S] = EigComp2SD_power_new(train_decomp,[],alpha);
%[S] = EigComp2SD_coef(train_decomp,[],alpha);
K = exp(-1*theta*S);

d = length(y);
if(size(y,1)==1)
    
    mask1 = repmat(y,d,1);
    mask2 = repmat(y',1,d);
else
    mask1 = repmat(y',d,1);
    mask2 = repmat(y,1,d);
end
K0 = double(mask1==mask2);
K0(K0 ==0) = -1;

if(strcmp(method,'ka')) % compute the objective with kernel alignment criterion
    ka = sum(sum(K0 .* K));
    k00 = sum(sum(K0 .* K0));
    kkk = sum(sum(K .* K));
    obj = -ka/sqrt((k00*kkk)) + lambda*norm(alpha - initial_alpha);
elseif(strcmp(method,'cs')) % compute the objective with class seperabiliy criterion
obj = -class_seperabiliy(K,y) + lambda*norm(alpha - initial_alpha);
end
end



function cs = class_seperabiliy(K,label)
n = size(K,1);
ulabel = unique(label);
nlabel = length(ulabel);

for ilabel = 1:nlabel
    ni(ilabel) = sum(label == ulabel(ilabel));
end
SB = 0;
SW = 0;
ST = 0;
for ilabel = 1:nlabel
    Ki = K(label==ulabel(ilabel),label==ulabel(ilabel));
    SB = SB + sum(Ki(:))/ni(ilabel);
end
SW = trace(K) - SB;
SB = SB - sum(K(:))/n;
ST = SW + SB;
cs = SB/ST;
end
