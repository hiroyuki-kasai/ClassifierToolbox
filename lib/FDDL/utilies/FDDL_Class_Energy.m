function [gap] = FDDL_Class_Energy(Ai,D,Xi,Xa,drls,trls,index,lambda1,lambda2,lambda3,lambda4,classn,tau2,tau3)
% ========================================================================
% Class energy computation of FDDL, Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% -----------------------------------------------------------------------
%   
% Input :   (1) Ai :  the data matrix of this class
%           (2) D :   the whole dictionary
%           (3) Xi:   the coefficient matrix of this class
%           (4) Xa:   the coefficient matrix of the whole class
%           (5) drls: labels of dictionary's column
%           (6) trls: labels of training samples
%           (7) index: label of class being processed
%           (8) lambda1 : parameter of l1-norm energy of coefficient
%           (9) lambda2 : parameter of within-class scatter
%           (10) lambda3 : parameter of between-class scatter
%           (11) lambda4:  parameter of l2-norm energy of coefficient
%           (12) classn:   the number of class
%           (13) tau2: parameter of ||A_i-D_iX_i^i||_F^2 in fidelity term
%           (14) tau3: parameter of ||D_jX_i^j||_F^2 in  fidelity term
% 
% Outputs : (1) gap  :    the total energy of some class
%
%------------------------------------------------------------------------

gap3  =   0;
gap4  =   0;
GAP1  =   norm((Ai-D*Xi),'fro')^2;% only for one class, no effect
GAP2  =   lambda1*sum(abs(Xi(:)));%
    
Xa(:,trls==index)  =  Xi;
tem_XA             =  Xa;
mean_xa            =  mean(tem_XA,2);
    
    n_c                =  size(Xi,2);
    for i_c = 1:classn
        t_X_ic   = tem_XA(:,trls==i_c);
        n_ic                =  size(t_X_ic,2);
        mean_xic = mean(t_X_ic,2);
%         gap3 = gap3+norm(t_X_ic-repmat(mean(t_X_ic,2),[1 size(t_X_ic,2)]),'fro')^2;
        gap4 = gap4+n_ic*(mean_xic-mean_xa)'*(mean_xic-mean_xa);
    end
    
    gap3 = norm(Xi-repmat(mean(Xi,2),[1 n_c]),'fro')^2;
    
    GAP3 = lambda2*gap3-lambda3*gap4;
    
    GAP4  =   lambda4*norm(Xi,'fro')^2;% only for one class, no effect
    
    GAP5  =   (tau2)*norm((Ai-D(:,drls==index)*Xi(drls==index,:)),'fro')^2;% only for one class, no effect
    GAP6  =   0;
    for i = 1:classn
        if i~=index
        GAP6 = GAP6+tau3*norm(D(:,drls==i)*Xi(drls==i,:),'fro')^2;
        end
    end
    
    gap = GAP1+GAP2+GAP3+GAP5+GAP6+GAP4;