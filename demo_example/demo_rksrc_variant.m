% demo.m 

close all;
clear;
clc;


%% load data
if 0
    load('../dataset/ORL_Face_img_cov.mat');

    train_num = length(TrainSet.y);
    test_num = length(TestSet.y);
    
    class_num = length(unique(TestSet.y));

    dim = size(TrainSet.GRCM2{1,1},1);
    TrainSet.X_cov = zeros(dim, dim, train_num);
    for i = 1 : train_num
        TrainSet.X_cov(:,:,i) = double(TrainSet.GRCM2{1,i});
    end

    TestSet.X_cov = zeros(dim, dim, test_num);
    for i = 1 : test_num
        TestSet.X_cov(:,:,i) = double(TestSet.GRCM2{1,i});
    end
else

    load('../dataset/toy_data_eccv12.mat');
    TrainSet.X_cov = TrainSet.X;
    TestSet.X_cov = TestSet.X;
    
    class_num = length(unique(TestSet.y));
end


%% R-KSRC (Stein kernel)
clear options;
options.mode = 'src'; % 'src', 'ip_linear', 'ip_max'
options.original_alpha = true;
options.theta = 0.01;
options.lambda = 1e-2;
options.verbose = true;
Accuracy_rksr_stein = rksr_classifier(TrainSet, TestSet, options);
fprintf('# R-KSRC (Stein): Accuracy = %5.5f\n', Accuracy_rksr_stein);

%% R-KSRC (Log Euclidean kernel)
clear options;
% The types of kernel:  
% 'Log-E poly.' -- Log-Euclidean polynomial kernel
% 'Log-E exp.' -- Log-Euclidean exponential kernel
% 'Log-E Gauss.' --Log-Euclidean Gaussian kernel
options.kernel = 'Log-E poly.' ; 
%options.mu = 0.01;
options.d = 43;
options.n = 50; % The degree of monomial in polynomial kernel or exponential kernel
options.Beta = 2e-2;% The parameter $\beta$ in the Gaussian kernel
options.SR_Lambda = 1e-3;% The regularizing coefficient in the objective of sparse coding
Accuracy_rksr_logeuc = LogEuc_SPD_SRC_Classification(TrainSet, TestSet, options);
fprintf('# R-KSRC (LogEuc): Accuracy = %5.5f\n', Accuracy_rksr_logeuc);


%% display accuracy
fprintf('\n\n## Summary of results\n\n')
fprintf('# R-KSRC (Stein): Accuracy = %5.5f\n', Accuracy_rksr_stein);
fprintf('# R-KSRC (LogEuc): Accuracy = %5.5f\n', Accuracy_rksr_logeuc);

