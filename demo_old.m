% demo.m 

close all;
clear;
clc;

%% set parameters
k = 5;


%% load data
load('./dataset/ORL_Face_img_cov.mat');


%% perform RCM k-NN classifier with 
% GRCM2 with eigenvalue-based distance
grcm_accuracy = rcm_knn_classifier(TrainSet, TestSet,'GRCM', '2', 'EV', k);
% RCM4 with eigenvalue-based distance
rcm_accuracy = rcm_knn_classifier(TrainSet, TestSet, 'RCM', '4', 'EV', k);

fprintf('\n');
fprintf('# GRCM2 Accuracy = %5.2f\n', grcm_accuracy);
fprintf('# RCM4 Accuracy = %5.2f\n', rcm_accuracy);