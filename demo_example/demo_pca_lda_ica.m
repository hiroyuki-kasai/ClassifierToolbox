% test.m

close all;
clear;
clc;

% set parameters
k = 3;
distance_type = 'Frobenius';
%distance_type = 'Angle';


% load dataset
%dataset = 'orl_face';
%dataset = 'YaleB_small_face';
dataset = 'usps';

if strcmp(dataset, 'orl_face')
    load('../dataset/ORL_Face_img.mat');
elseif strcmp(dataset, 'YaleB_small_face')
    load('../dataset/Cropped_YaleB_small_img_24x20.mat');
elseif strcmp(dataset, 'usps')    
    load('../dataset/USPS.mat');    
end




%% subtract mean from training set, test set
m = mean(TrainSet.X, 2);
TrainSet_normalized.X   = TrainSet.X - m * ones(1, size(TrainSet.X, 2));
TrainSet_normalized.y   = TrainSet.y;
TestSet_normalized.X    = TestSet.X - m * ones(1, size(TestSet.X, 2));
TestSet_normalized.y    = TestSet.y;

%  normalization_method = 'mean_std';
%  [TrainSet_normalized1.X, TrainSet_normalized1.y, TestSet_normalized1.X, TestSet_normalized1.y] = ...
%                  dataset_normalization(normalization_method, TrainSet.X, TrainSet.y, TestSet.X, TestSet.y);


%% PCA
fprintf('##  Algorithm : PCA ...\n');
W_pca = EigenfaceCore(TrainSet_normalized.X);
[P_train, P_test] = project_datasets(TrainSet_normalized, TestSet_normalized, W_pca);
accuracy_PCA = knn_classifier(P_train, P_test, k, distance_type);
fprintf('##  Accuracy = %5.2f\n\n', accuracy_PCA);


%% LDA
fprintf('##  Algorithm : LDA ...\n');
W_lda = FisherfaceCore(TrainSet_normalized.X, class_num);
[P_train, P_test] = project_datasets(TrainSet_normalized, TestSet_normalized, W_lda);
accuracy_LDA = knn_classifier(P_train, P_test, k, distance_type);
fprintf('##  Accuracy = %5.2f\n\n', accuracy_LDA);


%% ICA
fprintf('##  Algorithm : ICA ...\n');
dimension = size(TrainSet_normalized.X, 1);
if dimension >= train_num % a bit strange
    W_ica = fastica(TrainSet_normalized.X')';
else
    W_ica = fastica(TrainSet_normalized.X);
end
[P_train, P_test] = project_datasets(TrainSet_normalized, TestSet_normalized, W_ica);
accuracy_ICA = knn_classifier(P_train, P_test, k, distance_type);
fprintf('##  Accuracy = %5.2f %%\n', accuracy_ICA);


%% 
fprintf('\n\n## Summary of results\n\n')
fprintf('# PCA: Accuracy = %5.2f\n', accuracy_PCA);
fprintf('# LDA: Accuracy = %5.2f\n', accuracy_LDA);
fprintf('# ICA: Accuracy = %5.2f\n\n', accuracy_ICA);