function [Dict,Drls,CoefM,CMlabel] = FDDL(TrainDat,TrainLabel,opts)
% ========================================================================
% Fisher Discriminative Dictionary Learning (FDDL), Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
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
% This is an implementation of the algorithm for learning the
% Fisher Discriminative Dictionary from a labeled training data
%
% Please refer to the following paper
%
% Meng Yang, Lei Zhang, Xiangchu Feng, and David Zhang,"Fisher Discrimination 
% Dictionary Learning for Sparse Representation", In IEEE Int. Conf. on
% Computer Vision, 2011.
% 
%----------------------------------------------------------------------
%
%Input : (1) TrainDat: the training data matrix. 
%                      Each column is a training sample
%        (2) TrainDabel: the training data labels
%        (3) opts      : the struture of parameters
%               .nClass   the number of classes
%               .wayInit  the way to initialize the dictionary
%               .lambda1  the parameter of l1-norm energy of coefficient
%               .lambda2  the parameter of l2-norm of Fisher Discriminative
%               coefficient term
%               .nIter    the number of FDDL's iteration
%               .show     sign value of showing the gap sequence
%
%Output: (1) Dict:  the learnt dictionary via FDDL
%        (2) Drls:  the labels of learnt dictionary's columns
%        (2) CoefM: Mean Coefficient Matrix. Each column is a mean coef
%                   vector
%        (3) CMlabel: the labels of CoefM's columns.
%
%-----------------------------------------------------------------------
%
%Usage:
%Given a training data, including TrainDat and TrainLabel, and the
%parameters, opts.
%
%[Dict,CoefM,CMlabel] = FDDL(TrainDat,TrainLabel,opts)
%-----------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%
% normalize energy
%%%%%%%%%%%%%%%%%%
TrainDat = TrainDat*diag(1./sqrt(sum(TrainDat.*TrainDat)));

%%%%%%%%%%%%%%%%%%
%initialize dict
%%%%%%%%%%%%%%%%%%
Dict_ini  =  []; 
Dlabel_ini = [];
for ci = 1:opts.nClass
    cdat          =    TrainDat(:,TrainLabel==ci);
    dict          =    FDDL_INID(cdat,size(cdat,2),opts.wayInit);
    Dict_ini      =    [Dict_ini dict];
    Dlabel_ini    =    [Dlabel_ini repmat(ci,[1 size(dict,2)])];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialize coef without between-class scatter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ini_par.tau         =     opts.lambda1;
ini_par.lambda      =     opts.lambda2;
ini_ipts.D          =     Dict_ini;
coef = zeros(size(Dict_ini,2),size(TrainDat,2));
if size(Dict_ini,1)>size(Dict_ini,2)
      ini_par.c        =    1.05*eigs(Dict_ini'*Dict_ini,1);
else
      ini_par.c        =    1.05*eigs(Dict_ini*Dict_ini',1);
end
for ci =  1:opts.nClass
    fprintf(['Initializing Coef:  Class ' num2str(ci) '\n']);
    ini_ipts.X      =    TrainDat(:,TrainLabel==ci);
    [ini_opts]      =    FDDL_INIC (ini_ipts,ini_par);
    coef(:,TrainLabel ==ci) =    ini_opts.A;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Main loop of Fisher Discriminative Dictionary Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Fish_par.dls        =     Dlabel_ini;
 Fish_ipts.D         =     Dict_ini;
 Fish_ipts.trls      =     TrainLabel;
 Fish_par.tau        =     opts.lambda1;
 Fish_par.lambda2    =     opts.lambda2;
 
 Fish_nit            =     1;
 drls                =     Dlabel_ini;
 while Fish_nit<=opts.nIter  
    if size(Fish_ipts.D,1)>size(Fish_ipts.D,2)
      Fish_par.c        =    1.05*eigs(Fish_ipts.D'*Fish_ipts.D,1);
    else
      Fish_par.c        =    1.05*eigs(Fish_ipts.D*Fish_ipts.D',1);
    end
    %-------------------------
    %updating the coefficient
    %-------------------------
    for ci = 1:opts.nClass
        fprintf(['Updating coefficients, class: ' num2str(ci) '\n'])
        Fish_ipts.X         =  TrainDat(:,TrainLabel==ci);
        Fish_ipts.A         =  coef;
        Fish_par.index      =  ci; 
        [Copts]             =  FDDL_SpaCoef (Fish_ipts,Fish_par);
        coef(:,TrainLabel==ci)    =  Copts.A;
        CMlabel(ci)         =  ci;
        CoefM(:,ci)         =  mean(Copts.A,2);
    end
    [GAP_coding(Fish_nit)]  =  FDDL_FDL_Energy(TrainDat,coef,opts.nClass,Fish_par,Fish_ipts);
    
    %------------------------
    %updating the dictionary
    %------------------------
    for ci = 1:opts.nClass
        fprintf(['Updating dictionary, class: ' num2str(ci) '\n']);     
        [Fish_ipts.D(:,drls==ci),Delt(ci).delet]= FDDL_UpdateDi (TrainDat,coef,...
            ci,opts.nClass,Fish_ipts,Fish_par);
    end
    [GAP_dict(Fish_nit)]  =  FDDL_FDL_Energy(TrainDat,coef,opts.nClass,Fish_par,Fish_ipts);
    
    newD = []; newdrls = []; newcoef = [];
    for ci = 1:opts.nClass
        delet = Delt(ci).delet;
        if isempty(delet)
           classD = Fish_ipts.D(:,drls==ci);
           newD = [newD classD];
           newdrls = [newdrls repmat(ci,[1 size(classD,2)])];
           newcoef = [newcoef; coef(drls==ci,:)];
        else
            temp = Fish_ipts.D(:,drls==ci);
            temp_coef = coef(drls==ci,:);
            for temp_i = 1:size(temp,2)
                if sum(delet==temp_i)==0
                    newD = [newD temp(:,temp_i)];
                    newdrls = [newdrls ci];
                    newcoef = [newcoef;temp_coef(temp_i,:)];
                end
            end
        end
    end
    
    Fish_ipts.D  = newD;
    coef         = newcoef;
    drls         = newdrls;
    Fish_par.dls        =     drls;
    
    Fish_nit = Fish_nit +1;
    end
    
    Dict = Fish_ipts.D;
    Drls = drls;
    
    if opts.show
    subplot(1,2,1);plot(GAP_coding,'-*');title('GAP_coding');
    subplot(1,2,2);plot(GAP_dict,'-o');title('GAP_dict'); 
    end
return;