function [gap] = FDDL_FDL_Energy(Aa,Xa,classn,Fish_par,Fish_ipts)
% ========================================================================
% Total energy computation of FDDL, Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% -----------------------------------------------------------------------
%   
% Input :   (1) Aa :  the data matrix of all class
%           (2) Xa:   the coefficient matrix of the whole class
%           (3) classn:   the number of class
%           (4) Fish_par
%                      .dls  labels of dictionary's column
%                      .tau  parameter of l1-norm energy of coefficient
%                      .lambda2  parameter of within-class scatter
%           (5) Fish_ipts
%                      .D       The dictioanry
%                      .trls    labels of training samples
% 
% Outputs : (1) gap  :    the total energy
%
%------------------------------------------------------------------------
 D  =  Fish_ipts.D;
 drls    = Fish_par.dls;
 trls    = Fish_ipts.trls;
 lambda1 = Fish_par.tau;
 lambda2 = Fish_par.lambda2;
 lambda3 = lambda2;
 lambda4 = lambda2;
 tau2    = 1;
 tau3    = 1;
 
 
    gap3  =   0;
    gap4  =   0;
    GAP1  =   norm((Aa-D*Xa),'fro')^2;% 
    GAP2  =   lambda1*sum(abs(Xa(:)));%
    tem_XA = Xa;
    for i_c = 1:classn
        t_X_ic  = tem_XA(:,trls==i_c);
        gap3 = gap3+norm(t_X_ic-repmat(mean(t_X_ic,2),[1 size(t_X_ic,2)]),'fro')^2;
        gap4 = gap4+size(t_X_ic,2)*(mean(t_X_ic,2)-mean(tem_XA,2))'*(mean(t_X_ic,2)-mean(tem_XA,2));
    end
      
    GAP3 = lambda2*gap3-lambda3*gap4;
    
    GAP4  =   lambda4*norm(Xa,'fro')^2;% 
    
    GAP6  =   0;
    GAP5  =   0;
    for i = 1:classn
        Ai = Aa(:,trls==i);
        Xi = Xa(:,trls==i);
        GAP5  = GAP5 + (tau2)*norm(Ai-D(:,drls==i)*Xi(drls==i,:),'fro')^2;% only for one class, no effect
        for j = 1:classn
            if j~=i
            GAP6 = GAP6+tau3*norm(D(:,drls==j)*Xi(drls==j,:),'fro')^2;
            end
        end
    end
    
    gap = GAP1+GAP2+GAP3+GAP5+GAP6+GAP4;