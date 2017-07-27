function [p,D,C] = JDDLDR(Tr_DAT,trls,nDim,numcomps,lambda1,lambda2,gamma1,gamma2,MaxIteration_num)
% Joint discriminative dimensionality reduction and dictionary learning (JDDLDR), Version 1.0
% Copyright(c) 2013  Meng YANG, Zhizhao Feng, Lei Zhang, Yan Liu, David Zhang
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
% 
%----------------------------------------------------------------------
%
%Input : (1) Tr_DAT:   the training data matrix. 
%                      Each column is a training sample
%        (2) trls:     the training data labels
%        (3) nDim:     the dimensionality after learning
%        (4) numcomps: the number of samples in each class
%        (5) lambda1:  the parameter of l2-norm energy of coefficient
%        (6) lambda2:  the parameter of within-class scatter energy of coef
%        (7) gamma1:   the parameter of ||PAt||
%        (8) gamma2:   the parameter of ||PAb||
%        (9) MaxIteration_num:  the number of iterations
%
%Output: (1) p:  The dimension reduction matrix
%        (2) D:  the dictionary
%        (2) C:  coding coefficient matrix
%
%-----------------------------------------------------------------------
%


nClass      =  max(trls);                 % number of class in the database (100 for AR)
MaxIter_DC  =  10;
MaxIter_P   =  10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda_a =  lambda1;
lambda_b =  lambda2;

Mean_Image = mean(Tr_DAT,2);
ZA = Tr_DAT-Mean_Image*ones(1,size(Tr_DAT,2));
ZB = [];
for class=1:nClass
    Class_im   =   Tr_DAT(:,(trls==class)); % learn by each class     
    ZB(:,(trls==class))=repmat(mean(Class_im,2)-Mean_Image,1,size(Class_im,2));
end

% [p,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,nDim); % learn a PCA projection matrix p, the dimension is set by par.nDim 
[p,Eigen_Value]=Find_K_Max_Gen_Eigen(ZA*ZA'+ZB*ZB',eye(size(Tr_DAT,1)),nDim);
    
iter_num_main = 1; % main iteration times  

% initializing D and C
[D,C]=JDDLDR_DCinit(p'*Tr_DAT,numcomps,trls,lambda_a);

while iter_num_main<MaxIteration_num % the main iteration, to update p and D
    
    fprintf('%d: ', iter_num_main);
    % update D and C;
    [D,C]=JDDLDR_UDC(p'*Tr_DAT,trls,lambda_a,lambda_b,D,C,MaxIter_DC);

    % update P
    p = JDDLDR_UP3(Tr_DAT,p,ZA,ZB,D,C,gamma1,gamma2,MaxIter_P);
    
    iter_num_main = iter_num_main + 1;
end