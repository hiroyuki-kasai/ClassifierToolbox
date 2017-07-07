% demo.m 

close all;
clear;
clc;

%% set parameters




%% load data
if 1
    load('../dataset/ORL_Face_img_cov.mat');

    train_num = length(TrainSet.y);
    test_num = length(TestSet.y);

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

    load('../dataset/toy_data_iccv12.mat');
    TrainSet.X_cov = TrainSet.X;
    TestSet.X_cov = TestSet.X;
end


%% perform

% RSR Classifier
options.mode = 'src'; % 'src', 'ip_linear', 'ip_max'
options.original_alpha = true;
options.theta = 0.01;
options.lambda = 1e-2;
options.verbose = true;
Accuracy_rsr = rsr_classifier(TrainSet, TestSet, options);


fprintf('\n');
fprintf('# RSR Accuracy = %5.5f\n', Accuracy_rsr);
