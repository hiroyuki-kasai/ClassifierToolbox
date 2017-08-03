close all;
clear;
clc;


%% load dataset
%load('../dataset/ORL_Face_img.mat');
%load('../dataset/Brodatz_texture_img_small_set.mat');
%load('../dataset/AR_Face_img_27x20.mat'); 
%load('../dataset/AR_Face_img_60x43.mat'); 
load('../dataset/USPS.mat'); 
%load('../dataset/MNIST.mat');
%load('../dataset/COIL20.mat');
%load('../dataset/COIL100.mat');



%% reduce dataset for a quick test if necessary
max_class_num = 10;
max_train_samples = 2;
max_test_samples = 15;
[TrainSet, TestSet, train_num, test_num, class_num] = reduce_dataset(TrainSet, TestSet, max_class_num, max_train_samples, max_test_samples);


%% set paramters
eigenface_flag = true;
eigenface_dim = floor(train_num/2); % example
dim = size(TrainSet.X, 1);
if eigenface_dim > dim
    eigenface_dim = dim;
end


%% normalize dataset
[TrainSet_normalized.X, TrainSet_normalized.y] = data_normalization(TrainSet.X, TrainSet.y, 'std');        
[TestSet_normalized.X, TestSet_normalized.y] = data_normalization(TestSet.X, TestSet.y, 'std');     


%% SVM
options.verbose = true;
options.eigenface = eigenface_flag;
options.eigenface_dim = eigenface_dim;
accuracy_svm = svm_classifier(TrainSet, TestSet, train_num, test_num, class_num, options);
fprintf('# SVM: Accuracy = %5.5f\n', accuracy_svm);


%% LSR
lambda = 0.001;
options.verbose = true;
[accuracy_lsr, ~, ~] = lsr(TrainSet, TestSet, train_num, test_num, class_num, lambda, options);
fprintf('# LSR: Accuracy = %5.5f\n', accuracy_lsr);


%% LRC
clear options;
options.verbose = true;
accuracy_lrc = lrc(TrainSet_normalized, TestSet_normalized, test_num, class_num, options);
fprintf('# LRC: Accuracy = %5.5f\n', accuracy_lrc);


%% LDRC
clear options;
options.verbose = true;
accuracy_ldrc = ldrc(TrainSet_normalized, TestSet_normalized, train_num, test_num, class_num, eigenface_dim, options);
fprintf('# LDRC: Accuracy = %5.5f\n', accuracy_ldrc);


%% LCDRC
clear options;
options.verbose = true;
accuracy_lcdrc = lcdrc(TrainSet_normalized, TestSet_normalized, train_num, test_num, class_num, eigenface_dim, options);
fprintf('# LCDRC: Accuracy = %5.5f\n', accuracy_lcdrc);





%% display accuracy
fprintf('\n\n## Summary of results\n\n')
fprintf('# SVM: Accuracy = %5.5f\n', accuracy_svm);
fprintf('# LSR: Accuracy = %5.5f\n', accuracy_lsr);
fprintf('# LRC: Accuracy = %5.5f\n', accuracy_lrc);
fprintf('# LDRC: Accuracy = %5.5f\n', accuracy_ldrc);
fprintf('# LCDRC: Accuracy = %5.5f\n', accuracy_lcdrc);
