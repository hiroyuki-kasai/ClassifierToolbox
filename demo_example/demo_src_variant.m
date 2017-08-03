close all;
clear;
clc;


%% load dataset
%load('../dataset/ORL_Face_img.mat');
%load('../dataset/AR_Face_img_27x20.mat'); 
%load('../dataset/AR_Face_img_60x43.mat'); 
%load('../dataset/USPS.mat'); 
load('../dataset/MNIST.mat'); % become worse
%load('../dataset/COIL20.mat');
%load('../dataset/COIL100.mat');


%% reduce dataset for a quick test if necessary
max_class_num = 10;
max_train_samples = 5;
max_test_samples = 15;
[TrainSet, TestSet, train_num, test_num, class_num] = reduce_dataset(TrainSet, TestSet, max_class_num, max_train_samples, max_test_samples);


%% set paramters
verbose = true;
eigenface_flag = true;
eigenface_dim = floor(train_num/2); % example
dim = size(TrainSet.X, 1);
if eigenface_dim > dim
    eigenface_dim = dim;
end


%% normalize dataset
[TrainSet.X, TrainSet.y] = data_normalization(TrainSet.X, TrainSet.y, 'std');        
[TestSet.X, TestSet.y] = data_normalization(TestSet.X, TestSet.y, 'std');     


%% CRC
clear options;
lambda = 0.001;
options.verbose = true;
options.eigenface = eigenface_flag;
options.eigenface_dim = eigenface_dim;
accuracy_crc = crc(TrainSet, TestSet, train_num, test_num, class_num, lambda, options);
fprintf('# CRC: Accuracy = %5.5f\n', accuracy_crc);


%% SRC
clear options;
lambda = 0.001;
options.verbose = verbose;
options.eigenface = eigenface_flag;
options.eigenface_dim = eigenface_dim;
accuracy_src = src(TrainSet, TestSet, train_num, test_num, class_num, lambda, options);
fprintf('# SRC Accuracy = %5.5f\n', accuracy_src);


%% ESRC
clear options;
lambda = 0.001;
options.verbose = verbose;
options.eigenface = eigenface_flag;
options.eigenface_dim = eigenface_dim;
accuracy_esrc = esrc(TrainSet, TestSet, train_num, test_num, class_num, lambda, options);
fprintf('# ESRC Accuracy = %5.5f\n', accuracy_esrc);


%% SSRC
clear options;
lambda = 0.001;
options.verbose = verbose;
options.eigenface = eigenface_flag;
options.pca_dim = eigenface_dim;
accuracy_ssrc = ssrc(TrainSet, TestSet, train_num, test_num, class_num, lambda, options);
fprintf('# SSRC Accuracy = %5.5f\n', accuracy_ssrc);


%% SR-RLS
clear options;
options.verbose = verbose;
options.eigenface = eigenface_flag;
lambda_l1 = 0.001;
lambda_l2 = 0.001;
options.eigenface_dim = eigenface_dim;
accuracy_sr_rls = sr_rls(TrainSet, TestSet, train_num, test_num, class_num, lambda_l1, lambda_l2, options);
fprintf('# SR-RLS Accuracy = %5.5f\n', accuracy_sr_rls);


%% SDR_SLR
clear options;
v = 1*sqrt(size(TrainSet.X,1))^-1;
options.slr_v = v;
options.slr_lambda = 1;
options.slr_tau = 0.01;
options.slr_delta = 1.2 * v;
options.slr_eta = 1 * v;
% parameters setting for sdr
options.sdr_beta = 10;
options.sdr_gamma = 10;
options.verbose = verbose;
accuracy_sdr_slr = sdr_slr(TrainSet, TestSet, train_num, test_num, class_num, options);
fprintf('# SDR-SLR Accuracy = %5.5f\n', accuracy_sdr_slr);


%% RobustPCA-SRC
lambda = 0.001;
options.verbose = true;
options.eigenface = eigenface_flag;
options.eigenface_dim = eigenface_dim;
options.solver_max_iter = 5;
accuracy_robust_pca = rpca_src(TrainSet, TestSet, train_num, test_num, class_num, lambda, options);
fprintf('# RobustPCA-SRC: Accuracy = %5.5f\n', accuracy_robust_pca);



%% display accuracy
fprintf('\n\n## Summary of results\n\n')
fprintf('# CRC: Accuracy = %5.5f\n', accuracy_crc);
fprintf('# SRC: Accuracy = %5.5f\n', accuracy_src);
fprintf('# ESRC: Accuracy = %5.5f\n', accuracy_esrc);
fprintf('# SSRC Accuracy = %5.5f\n', accuracy_ssrc);
fprintf('# SR-RLS Accuracy = %5.5f\n', accuracy_sr_rls);
fprintf('# SDR-SLR Accuracy = %5.5f\n', accuracy_sdr_slr);
fprintf('# RobustPCA-SRC: Accuracy = %5.5f\n', accuracy_robust_pca);
