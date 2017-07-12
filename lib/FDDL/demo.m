close all;
clear all;
clc;

addpath([cd '/utilies']);
load(['AR_EigenFace']);

%%%%%%%%%%%%%%%%%%%%%%%%
%FDDL parameter
%%%%%%%%%%%%%%%%%%%%%%%%
opts.nClass        =   100;
opts.wayInit       =   'PCA';
opts.lambda1       =   0.005;
opts.lambda2       =   0.05;
opts.nIter         =   15;
opts.show          =   true;
[Dict,Drls,CoefM,CMlabel] = FDDL(tr_dat,trls,opts);

%%%%%%%%%%%%%%%%%%%%%%%%
% Sparse Classification
%%%%%%%%%%%%%%%%%%%%%%%%
lambda   =   0.005;
nClass   =   opts.nClass;
weight   =   0.5;

td1_ipts.D    =   Dict;
td1_ipts.tau1 =   lambda;
if size(td1_ipts.D,1)>=size(td1_ipts.D,2)
   td1_par.eigenv = eigs(td1_ipts.D'*td1_ipts.D,1);
else
   td1_par.eigenv = eigs(td1_ipts.D*td1_ipts.D',1);  
end

ID   =   [];
for indTest = 1:size(tt_dat,2)
    fprintf(['Totalnum:' num2str(size(tt_dat,2)) 'Nowprocess:' num2str(indTest) '\n']);
    td1_ipts.y          =      tt_dat(:,indTest);   
    [opts]              =      IPM_SC(td1_ipts,td1_par);
    s                   =      opts.x;
    
    for indClass  =  1:nClass
        temp_s            =  zeros(size(s));
        temp_s(indClass==Drls) = s(indClass==Drls);
        zz                =  tt_dat(:,indTest)-td1_ipts.D*temp_s;
        gap(indClass)     =  zz(:)'*zz(:);
        
        mean_coef_c         =   CoefM(:,indClass);
        gCoef3(indClass)    =  norm(s-mean_coef_c,2)^2;    
    end
    
    wgap3  = gap + weight*gCoef3;
    index3 = find(wgap3==min(wgap3));
    id3    = index3(1);
    ID     = [ID id3];
end  

fprintf('%s%8f\n','reco_rate  =  ',sum(ID==ttls)/(length(ttls)));