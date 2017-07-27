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

clear all;clc;
addpath('utilities');

numcomps = 3;      % tr_num = 3;
nDim     = 400;    % feature dimension
load(['FRGC/FRGC' num2str(numcomps) '_10.mat']);  % the 10th round
Train_DAT = Train_DAT./( repmat(sqrt(sum(Train_DAT.*Train_DAT)), [size(Train_DAT,1),1]) );
Test_DAT = Test_DAT./( repmat(sqrt(sum(Test_DAT.*Test_DAT)), [size(Test_DAT,1),1]) );

lambda1 = 0.05; % sparse
lambda2 = 0.05; % mean
gamma1  = 10;   % parameter of Eq.(1)
gamma2  = 1;    % parameter of Eq.(1)

% jointly learn the dictionary and the dimension reduction matrix
MaxIter  = 5;
[P,D,C] = JDDLDR(Train_DAT,trls,nDim,numcomps,lambda1,lambda2,gamma1,gamma2,MaxIter);

% prepare the dictionary and testing set
tr_dat  = []; trls = [];
for i = 1:size(D,2)
   tr_dat = [tr_dat D(i).M];
   trls   = [trls repmat(i,[1 size(D(i).M,2)])];
end

tt_dat  =  P'*single(Test_DAT);  clear Test_DAT;
tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [nDim,1]) );
tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [nDim,1]) );

% do collaborative representation based classification
lambda =  0.001;
correct_rate = Fun_CRC(tr_dat,trls,tt_dat,ttls,lambda);

fid = fopen(['result\demo_JDDLR_result_FRGC.txt'],'a');
fprintf(fid,'\n%s\n','==========================================');
fprintf(fid,'%s%8f\n','reco_rate1 = ',correct_rate);
fclose(fid);
