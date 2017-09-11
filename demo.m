% demo.m 

close all;
clear;
clc;


%% load dataset
load('./dataset/AR_Face_img_60x43.mat'); 


%% Set option
options.verbose = true;


% %% reduce dataset for a quick test if necessary
% max_class_num = 10;
% max_train_samples = 2;
% max_test_samples = 15;
% [TrainSet, TestSet, train_num, test_num, class_num] = reduce_dataset(TrainSet, TestSet, max_class_num, max_train_samples, max_test_samples);
% 
% 
% 
% %% normalize dataset if necessary
% [TrainSet.X, TrainSet.y] = data_normalization(TrainSet.X, TrainSet.y, 'std');        
% [TestSet.X, TestSet.y] = data_normalization(TestSet.X, TestSet.y, 'std');     


%% execute LSR
[accuracy_lsr, ~, ~] = lsr(TrainSet, TestSet, train_num, test_num, class_num, 0.001, options);


%% execute LRC
accuracy_lrc = lrc(TrainSet, TestSet, test_num, class_num, options);


%% display accuracy
fprintf('\n\n## Summary of results\n\n')
fprintf('# LSR: Accuracy = %5.5f\n', accuracy_lsr);
fprintf('# LRC: Accuracy = %5.5f\n', accuracy_lrc);

