close all;
clear;
clc;


%% load dataset
%load('../dataset/ORL_Face_img.mat');
%load('../dataset/Brodatz_texture_img_small_set.mat');
%load('../dataset/Cropped_YaleB_small_img_24x20.mat'); % does not work. Why???
%load('../dataset/AR_Face_img_27x20_from_LKDL.mat'); 
%load('../dataset/AR_Face_img_27x20_natura_downloaded.mat'); 
%load('../dataset/AR_Face_img_60x43_from_CRC.mat'); 
%load('../dataset/AR_Face_img_60x43_from_CRC.mat'); 
%load('../dataset/COIL100.mat');
load('../dataset/USPS.mat');
%load('../dataset/MNIST.mat');

ksvd_iter_num = 2;
dict_size = 3;
cardinality = 20;
max_class_num = 100;
max_train_samples = 5;
max_test_samples = 5;
[TrainSet, TestSet, train_num, test_num, class_num] = reduce_dataset(TrainSet, TestSet, max_class_num, max_train_samples, max_test_samples);


%% generate matrix-based labels
TrainSet.y_mat  = convert_labelvec_to_mat(TrainSet.y, train_num, class_num);
TestSet.y_mat   = convert_labelvec_to_mat(TestSet.y, test_num, class_num);           


%% subtract mean from training set, test set
normalization_method = 'std';
[TrainSet.X, TrainSet.y, TestSet.X, TestSet.y] = ...
                dataset_normalization(normalization_method, TrainSet.X, TrainSet.y, TestSet.X, TestSet.y);

            
            

    
%% KSVD Dictionary Learning
fprintf('#################### KSVD ####################\n');        
% KSVD 
params.num_classes         = class_num;
params.card                = cardinality; 
params.init_dic = 'partial_new';
params.alg_type = 'KSVD';
params.num_runs = 3;
params.iter = ksvd_iter_num;
params.dict_size = dict_size;
%rng('default');
[dic_cell, ~] = KSVD_trainer(params, double(TrainSet.X), TrainSet.y_mat);

% LC-KSVD-type Classifier
[Acc_KSVD_LC_KSVD_type, ~, ~] = KSVD_classifier(params, 'LC_KSVD_type', TestSet.X, TestSet.y_mat, TrainSet.X, TrainSet.y_mat, dic_cell);
fprintf('## KSVD: LC_KSVD type Accurary  = %5.2f\n', Acc_KSVD_LC_KSVD_type);        

% L_KSVD-type Classifier
[Acc_KSVD_LKDL_type, ~, ~] = KSVD_classifier(params, 'LKDL_type', TestSet.X, TestSet.y_mat, TrainSet.X, TrainSet.y_mat, dic_cell);
fprintf('## KSVD: L_KDL type Accurary  = %5.2f\n', Acc_KSVD_LKDL_type);    

fprintf('\n');             


%% LC-KSVD Dictionary Learning
fprintf('#################### LC-KSVD ####################\n');         
dictsize        = dict_size * class_num;
iter            = ksvd_iter_num;
iter4ini        = ksvd_iter_num;
sqrt_alpha      = 4;                % weights for label constraint term
sqrt_beta       = 2;                % weights for classification err term        

%% dictionary learning process
% get initial dictionary Dinit and Winit
%rng('default');
fprintf('## LC-KSVD initialization for LC-KSVD1&2 ...');
[Dinit, Tinit, Winit, Q_train] = initialization4LCKSVD(TrainSet.X, TrainSet.y_mat, dictsize, iter4ini, cardinality);
fprintf('done! \n');
[~, Acc_LC_KSVD_init_LKDL_type] = classification(Dinit, Winit, TestSet.X, TestSet.y_mat, cardinality);
fprintf('## LC-KSVD (init): LC_KSVD type Accurary  = %5.2f\n', Acc_LC_KSVD_init_LKDL_type); 


%% LC-KSVD1 (reconstruction err + class penalty)
fprintf('\n## Dictionary learning by LC-KSVD1 ...\n');
[D1, X1, T1, W1] = labelconsistentksvd1(TrainSet.X, Dinit, Q_train, Tinit, TrainSet.y_mat, iter, cardinality, sqrt_alpha);
[prediction1, Acc_LC_KSVD1_LKDL_type] = classification(D1, W1, TestSet.X, TestSet.y_mat, cardinality);
fprintf('## LC-KSVD1: LC_KSVD type Accurary  = %5.2f\n', Acc_LC_KSVD1_LKDL_type); 

%% LC-KSVD2 (reconstruction err + class penalty + classifier err)
fprintf('\n## Dictionary and classifier learning by LC-KSVD2 ...\n')
[D2,X2,T2,W2] = labelconsistentksvd2(TrainSet.X, Dinit, Q_train, Tinit, TrainSet.y_mat, Winit, iter, cardinality, sqrt_alpha, sqrt_beta);
[prediction2, Acc_LC_KSVD2_LKDL_type] = classification(D2, W2, TestSet.X, TestSet.y_mat, cardinality);
fprintf('## LC-KSVD2: LC_KSVD type Accurary  = %5.2f\n', Acc_LC_KSVD2_LKDL_type); 


fprintf('\n');  

%% JDDLDR Joint discriminative demensionality reduction and dictionary Learning
fprintf('#################### JDDLDR ####################\n');   
numcomps = dict_size;      % tr_num = 3;
nDim     = 400;    % feature dimension

if nDim > size(TrainSet.X, 1)
    nDim = floor(size(TrainSet.X, 1) * 0.5);
end

lambda1 = 0.05; % sparse
lambda2 = 0.05; % mean
gamma1  = 10;   % parameter of Eq.(1)
gamma2  = 1;    % parameter of Eq.(1)

% jointly learn the dictionary and the dimension reduction matrix
MaxIter  = 5;
[P,D,C] = JDDLDR(TrainSet.X, TrainSet.y, nDim, numcomps, lambda1, lambda2, gamma1, gamma2, MaxIter);

% prepare the dictionary and testing set
tr_dat  = []; trls = [];
for i = 1:size(D,2)
   tr_dat = [tr_dat D(i).M];
   trls   = [trls repmat(i,[1 size(D(i).M,2)])];
end

tt_dat  =  P'*single(TestSet.X);  %clear Test_DAT;
tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [nDim,1]) );
tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [nDim,1]) );

% do collaborative representation based classification
lambda =  0.001;
Acc_JDDLDR = Fun_CRC(tr_dat,trls,tt_dat,TestSet.y,lambda);
fprintf('## JDDLDR: Accurary  = %5.2f\n', Acc_JDDLDR); 


%% 
fprintf('\n\n## Summary of results\n\n')
fprintf('# KSVD: LC_KSVD type : Accuracy = %5.2f\n', Acc_KSVD_LC_KSVD_type);
fprintf('# KSVD: L_KDL type : Accuracy = %5.2f\n', Acc_KSVD_LKDL_type);
fprintf('# LC-KSVD(init): LC_KSVD type : Accuracy = %5.2f\n', Acc_LC_KSVD_init_LKDL_type);
fprintf('# LC-KSVD1: LC_KSVD type : Accuracy = %5.2f\n', Acc_LC_KSVD1_LKDL_type);
fprintf('# LC-KSVD2: LC_KSVD type Accurary = %5.2f\n', Acc_LC_KSVD2_LKDL_type);
fprintf('# JDDLDR: Accurary  = %5.2f\n', Acc_JDDLDR); 
