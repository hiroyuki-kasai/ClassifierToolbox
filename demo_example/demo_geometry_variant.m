% demo_geometry_variant.m 

close all;
clear;
clc;

verbose = 2;
lambda = 0.01;
normalization = false;


%% load data
load('../dataset/test_cov.mat');
TrainSet.X = TrainSet.X_cov;
TestSet.X = TestSet.X_cov;  


%% NN-AIRM
metric = 1;
Accuracy_nn_airm  = spd_knn_classifier(TrainSet, TestSet, metric);
fprintf('# NN (AIRM): Accuracy = %5.5f\n', Accuracy_nn_airm);


%% NN-S
metric = 2;
Accuracy_nn_stein  = spd_knn_classifier(TrainSet, TestSet, metric);
fprintf('# NN (Stein): Accuracy = %5.5f\n', Accuracy_nn_stein);


%% R-SRC (AIRM)
clear options;
options.verbose = verbose;
options.lambda = lambda;
options.normalization = normalization; 
Accuracy_rsrc = rsrc_classifier(TrainSet, TestSet, 'RSRC', options);
fprintf('# R-SRC: Accuracy = %5.5f\n', Accuracy_rsrc.residual);


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
options.d = size(TrainSet.X, 1);
options.n = 50; % The degree of monomial in polynomial kernel or exponential kernel
options.Beta = 2e-2;% The parameter $\beta$ in the Gaussian kernel
options.SR_Lambda = 1e-3;% The regularizing coefficient in the objective of sparse coding
Accuracy_rksr_logeuc = LogEuc_SPD_SRC_Classification(TrainSet, TestSet, options);
fprintf('# R-KSRC (LogEuc): Accuracy = %5.5f\n', Accuracy_rksr_logeuc);

 
newDim = floor(size(TrainSet.X, 1)/2);

%% NN-AIRM-DR (NN on dimensionaliy reduced (AIRM-based) SPD)
clear options;
options.verbose = 2;
options.maxiter = 5;
metric = 1;
[DR_TrainSet, DR_TestSet] = spd_dr(TrainSet, TestSet, newDim, metric, options);
Accuracy_nn_airm_dr   = spd_knn_classifier(DR_TrainSet, DR_TestSet, metric);
fprintf('# NN on low-dimensional SPD (AIRM): Accuracy = %5.5f\n', Accuracy_nn_airm_dr);


%% R-SRC-AIRM-DR (R-SRC on dimensionaliy reduced (AIRM-based) SPD)
clear options;
options.verbose = verbose;
options.lambda = lambda;
options.normalization = normalization;
DR_TrainSet.X_cov = DR_TrainSet.X;
DR_TestSet.X_cov = DR_TestSet.X;   
Accuracy_rsrc_airm_dr = rsrc_classifier(DR_TrainSet, DR_TestSet, 'RSRC', options);
fprintf('# R-SRC on low-dimensional SPD (AIRM): Accuracy = %5.5f\n', Accuracy_rsrc_airm_dr.residual);


%% R-KSRC-AIRM-DR (R-KSRC on dimensionaliy reduced (S divergence-based) SPD)
clear options;
options.mode = 'src'; % 'src', 'ip_linear', 'ip_max'
options.original_alpha = true;
options.theta = 0.01;
options.lambda = 1e-2;
options.verbose = true;
DR_TrainSet.X_cov = DR_TrainSet.X;
DR_TestSet.X_cov = DR_TestSet.X; 
Accuracy_rksrc_airm_dr = rksr_classifier(DR_TrainSet, DR_TestSet, options);
fprintf('# R-KSRC on low-dimensional SPD (AIRM): Accuracy = %5.5f\n', Accuracy_rksrc_airm_dr);


%% NN-S-DR (NN on dimensionaliy reduced (S divergence-based) SPD)
clear options;
options.verbose = 2;
options.maxiter = 5;
metric = 2;
[DR_TrainSet, DR_TestSet] = spd_dr(TrainSet, TestSet, newDim, metric, options);
Accuracy_nn_stein_dr   = spd_knn_classifier(DR_TrainSet, DR_TestSet, metric);   
fprintf('# NN on low-dimensional SPD (Stein): Accuracy = %5.5f\n', Accuracy_nn_stein_dr);


%% R-SRC-S-DR (R-SRC on dimensionaliy reduced (S divergence-based) SPD)
clear options;
options.verbose = verbose;
options.lambda = lambda;
options.normalization = normalization;
DR_TrainSet.X_cov = DR_TrainSet.X;
DR_TestSet.X_cov = DR_TestSet.X;   
Accuracy_rsrc_stein_dr = rsrc_classifier(DR_TrainSet, DR_TestSet, 'RSRC', options);
fprintf('# R-SRC on low-dimensional SPD (Stein): Accuracy = %5.5f\n', Accuracy_rsrc_stein_dr.residual);


%% R-KSRC-S-DR (R-KSRC on dimensionaliy reduced (S divergence-based) SPD)
clear options;
options.mode = 'src'; % 'src', 'ip_linear', 'ip_max'
options.original_alpha = true;
options.theta = 0.01;
options.lambda = 1e-2;
options.verbose = true;
DR_TrainSet.X_cov = DR_TrainSet.X;
DR_TestSet.X_cov = DR_TestSet.X; 
Accuracy_rksrc_stein_dr = rksr_classifier(DR_TrainSet, DR_TestSet, options);
fprintf('# R-KSRC on low-dimensional SPD (Stein): Accuracy = %5.5f\n', Accuracy_rksrc_stein_dr);

    
 
%% display accuracy
fprintf('\n\n## Summary of results\n\n')
fprintf('***** NN (nearest neighbor) classifier *****\n')
fprintf('# NN (AIRM): Accuracy = %5.5f\n', Accuracy_nn_airm);
fprintf('# NN (Stein): Accuracy = %5.5f\n', Accuracy_nn_stein);

fprintf('***** Geometry-based SRC classifier *****\n')
fprintf('# R-SRC: Accuracy = %5.5f\n', Accuracy_rsrc.residual);
fprintf('# R-KSRC (Stein): Accuracy = %5.5f\n', Accuracy_rksr_stein);
fprintf('# R-KSRC (LogEuc): Accuracy = %5.5f\n', Accuracy_rksr_logeuc);

fprintf('***** Geometry-aware dimensionality reduction + NN/SRC classifier *****\n')
fprintf('# NN on low-dimensional SPD (AIRM): Accuracy = %5.5f\n', Accuracy_nn_airm_dr);
fprintf('# R-SRC on low-dimensional SPD (AIRM): Accuracy = %5.5f\n', Accuracy_rsrc_airm_dr.residual);
fprintf('# R-KSRC on low-dimensional SPD (AIRM): Accuracy = %5.5f\n', Accuracy_rksrc_airm_dr);
fprintf('# NN on low-dimensional SPD (Stein): Accuracy = %5.5f\n', Accuracy_nn_stein_dr);
fprintf('# R-SRC on low-dimensional SPD (Stein): Accuracy = %5.5f\n', Accuracy_rsrc_stein_dr.residual);
fprintf('# R-KSRC on low-dimensional SPD (Stein): Accuracy = %5.5f\n', Accuracy_rksrc_stein_dr);

