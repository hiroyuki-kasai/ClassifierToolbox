%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jianjia Zhang, jz163@uowmail.edu.au Dec, 2014, all rights reserved
% For implementation details, please refer to:
% "Learning Discriminative Stein Kernel for SPD Matrices and Its Applications."
% arXiv preprint arXiv:1407.1974 (2014).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;
%% parameter settings
K = 5; %K of knn classifier, it is user provided
lambda = 0.01;% lembda is user provided.
%%
%load data, please modify it to use your own data
load('example_data.mat','train_data','train_label','test_data','test_label');

%% parameter settings
%original Stein kernel
opt.theta = 1; % a parameter in Stein kernel

opt.original_alpha = 1; % set to 1 to use original Stein kernel or 0 to use DSK
[test_kernel,train_kernel] = DSK_optimization(train_data,train_label,test_data,opt); % do DSK adjustment
[accu_SK] = kernel_knn_classification(test_kernel,train_label,K,test_label); % do knn classification

%% DSK with kernel alignment
opt.obj_method = 'ka'; % use kernel alignment criterion or 'cs' to use class seperabiliy criterion.
opt.lambda = lambda;
opt.original_alpha = 0;
[test_kernel_ka,train_kernel,optimal_alpha_ka] = DSK_optimization(train_data,train_label,test_data,opt);
[accu_DSK_ka] = kernel_knn_classification(test_kernel_ka,train_label,K,test_label);

%% DSK with class seperabiliy
opt.obj_method = 'cs'; % use kernel alignment criterion or 'cs' to use class seperabiliy criterion.
opt.lambda = lambda;
opt.original_alpha = 0;
[test_kernel_cs,train_kernel,optimal_alpha_cs] = DSK_optimization(train_data,train_label,test_data,opt);
[accu_DSK_cs] = kernel_knn_classification(test_kernel_cs,train_label,K,test_label);


%% Plotting
fprintf('The accuracy of SK is %5f\n', accu_SK);

fprintf('The accuracy of DSK with kernel alignment is %5f\n', accu_DSK_ka);
fprintf('The alpha in DSK is:\n');
disp(optimal_alpha_ka);

fprintf('The accuracy of DSK with class seperabiliy is %5f\n', accu_DSK_cs);
fprintf('The alpha in DSK is:\n');
disp(optimal_alpha_cs);
