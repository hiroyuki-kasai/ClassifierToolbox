% demo_rksrc_sdk.m 

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

    load('../dataset/test_cov.mat');
        
    class_num = length(unique(TestSet.y));
end


%% R-KSRC
% R-KSRC Classifier
options.mode = 'src'; % 'src', 'ip_linear', 'ip_max'
options.original_alpha = true;
options.theta = 0.01;
options.lambda = 1e-2;
options.verbose = true;
Accuracy_rsr = rksr_classifier(TrainSet, TestSet, options);
fprintf('# R-KSRC Accuracy = %5.5f\n', Accuracy_rsr);

% R-KSRC-DSK (kernel alignment) 
fprintf('# R-KSRC with DSK (KA) Classification ... ');
options.mode = 'src'; % 'src', 'ip_linear', 'ip_max'
options.theta = 0.01;
options.obj_method = 'ka'; % use kernel alignment criterion.
options.lambda = 1e-2;
options.original_alpha = 0;
options.verbose = true;
Accuracy_rsr_dsk_ka = rksr_dsk_classifier(TrainSet, TestSet, options);
fprintf('Accuracy = %5.5f\n', Accuracy_rsr_dsk_ka); 

% R-KSRC-DSK (class seperabiliy) 
fprintf('# R-KSRC with DSK (CS) Classification ... ');
options.mode = 'src'; % 'src', 'ip_linear', 'ip_max'
options.theta = 0.01;
options.obj_method = 'cs';
options.lambda = 1e-2;
options.original_alpha = 0;
options.verbose = true;
Accuracy_rsr_dsk_cs = rksr_dsk_classifier(TrainSet, TestSet, options);
fprintf('Accuracy = %5.5f\n', Accuracy_rsr_dsk_cs);       


%% K-NN with SK or DSK (TAPI2015)
% K-NN with SK (Stein Kernel)
fprintf('# K-NN with SK Classification ... ');
clear options;
options.theta = 0.01;
options.original_alpha = 1;
[test_kernel, train_kernel] = DSK_optimization_new(TrainSet,TestSet,options);
clear options;
options.verbose = true;
Accuracy_knn = kernel_knn_classification_new(test_kernel, TrainSet.y, class_num, TestSet.y, options); % do knn classification 
fprintf('Accuracy = %5.5f\n', Accuracy_knn);

% K-NN with DSK (Discriminative Stein Kernel)
fprintf('# K-NN with DSK (KA) Classification ... ');
clear options
options.theta = 0.01;
options.obj_method = 'ka';
options.lambda = 0.01;
options.original_alpha = 0;
[test_kernel,train_kernel,optimal_alpha] = DSK_optimization_new(TrainSet,TestSet,options);
clear options;
options.verbose = true;
Accuracy_knn_dsk_ka = kernel_knn_classification_new(test_kernel, TrainSet.y, class_num, TestSet.y,options);
fprintf('Accuracy = %5.5f\n', Accuracy_knn_dsk_ka);    


% K-NN with DSK (Discriminative Stein Kernel)
fprintf('# K-NN with DSK (CS) Classification ... ');
clear options
options.theta = 0.01;
options.obj_method = 'cs';
options.lambda = 0.01;
options.original_alpha = 0;
[test_kernel,train_kernel,optimal_alpha] = DSK_optimization_new(TrainSet,TestSet,options);
clear options;
options.verbose = true;
Accuracy_knn_dsk_cs = kernel_knn_classification_new(test_kernel,TrainSet.y,class_num,TestSet.y,options);
fprintf('Accuracy = %5.5f\n', Accuracy_knn_dsk_cs);



%% display accuracy
fprintf('\n\n## Summary of results\n\n')
fprintf('# R-KSRC: Accuracy = %5.5f\n', Accuracy_rsr);
fprintf('# R-KSRC-DSK-KA: Accuracy = %5.5f\n', Accuracy_rsr_dsk_ka); 
fprintf('# R-KSRC-DSK-CS: Accuracy = %5.5f\n', Accuracy_rsr_dsk_cs);
fprintf('# kNN: Accuracy = %5.5f\n', Accuracy_knn);
fprintf('# kNN-DSK-KA: Accuracy = %5.5f\n', Accuracy_knn_dsk_ka); 
fprintf('# kNN-DSK-CS: Accuracy = %5.5f\n', Accuracy_knn_dsk_cs);

