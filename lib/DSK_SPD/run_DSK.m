%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jianjia Zhang, jz163@uowmail.edu.au Dec, 2014, all rights reserved
% For implementation details, please refer to:
% "Learning Discriminative Stein Kernel for SPD Matrices and Its Applications."
% arXiv preprint arXiv:1407.1974 (2014).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
%% parameter settings
K = 2; %K of knn classifier, it is user provided
lambda = 0.01;% lembda is user provided.
%%
%load data, please modify it to use your own data
load('example_data.mat','train_data','train_label','test_data','test_label');
%% parameter settings
%original Stein kernel
opt.theta = 1; % a parameter in Stein kernel
opt.obj_method = 'ka'; % use kernel alignment criterion or 'cs' to use class seperabiliy criterion.
opt.original_alpha = 1; % set to 1 to use original Stein kernel or 0 to use DSK
[test_kernel,train_kernel] = DSK_optimization(train_data,train_label,test_data,opt); % do DSK adjustment
[accu_SK] = kernel_knn_classification(test_kernel,train_label,K,test_label); % do knn classification
%% DSK
opt.lambda = lambda;
opt.original_alpha = 0;
[test_kernel,train_kernel,optimal_alpha] = DSK_optimization(train_data,train_label,test_data,opt);
[accu_DSK] = kernel_knn_classification(test_kernel,train_label,K,test_label);
fprintf('The accuracy of SK is %5f\n', accu_SK);
fprintf('The accuracy of DSK is %5f\n', accu_DSK);
fprintf('The alpha in DSK is:\n');
disp(optimal_alpha);
