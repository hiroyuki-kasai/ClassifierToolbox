% demo.m 

close all;
clear;
clc;


%% load dataset

load('./dataset/AR_Face_img_60x43.mat'); 


%% reduce dataset for a quick test if necessary
max_class_num = 10;
max_train_samples = 2;
max_test_samples = 15;
[TrainSet, TestSet, train_num, test_num, class_num] = reduce_dataset(TrainSet, TestSet, max_class_num, max_train_samples, max_test_samples);



%% normalize dataset
[TrainSet.X, TrainSet.y] = data_normalization(TrainSet.X, TrainSet.y, 'std');        
[TestSet.X, TestSet.y] = data_normalization(TestSet.X, TestSet.y, 'std');     


%% LSR
lambda = 0.001;
options.verbose = true;
[accuracy_lsr, ~, ~] = lsr(TrainSet, TestSet, train_num, test_num, class_num, lambda, options);
fprintf('# LSR: Accuracy = %5.5f\n', accuracy_lsr);


%% LRC
clear options;
options.verbose = true;
accuracy_lrc = lrc(TrainSet, TestSet, test_num, class_num, options);
fprintf('# LRC: Accuracy = %5.5f\n', accuracy_lrc);



%% display accuracy
fprintf('\n\n## Summary of results\n\n')
fprintf('# LSR: Accuracy = %5.5f\n', accuracy_lsr);
fprintf('# LRC: Accuracy = %5.5f\n', accuracy_lrc);

