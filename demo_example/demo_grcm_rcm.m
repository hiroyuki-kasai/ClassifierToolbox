% demo_grcm_rcm.m

close all;
clear;
clc;


%% set parameters
k = 1;


%% load data
load('../dataset/ORL_Face_img_cov.mat');


%% perform GRCM k-NN classifier 
grcm_ev_accuracy = rcm_knn_classifier(TrainSet, TestSet,'GRCM', '2', 'EV', k);
grcm_ev2_accuracy = rcm_knn_classifier(TrainSet, TestSet,'GRCM', '2', 'EV2', k);
grcm_stein_accuracy = rcm_knn_classifier(TrainSet, TestSet,'GRCM', '2', 'Stein', k);
grcm_airm_accuracy = rcm_knn_classifier(TrainSet, TestSet,'GRCM', '2', 'AIRM', k);
grcm_lerm_accuracy = rcm_knn_classifier(TrainSet, TestSet,'GRCM', '2', 'LERM', k);
grcm_frobenius_accuracy = rcm_knn_classifier(TrainSet, TestSet,'GRCM', '2', 'Frobenius', k);


%% perform RCM k-NN classifier with 
rcm_ev_accuracy = rcm_knn_classifier(TrainSet, TestSet,'RCM', '4', 'EV', k);
rcm_ev2_accuracy = rcm_knn_classifier(TrainSet, TestSet,'RCM', '4', 'EV2', k);
rcm_stein_accuracy = rcm_knn_classifier(TrainSet, TestSet,'RCM', '4', 'Stein', k);
rcm_airm_accuracy = rcm_knn_classifier(TrainSet, TestSet,'RCM', '4', 'AIRM', k);
rcm_lerm_accuracy = rcm_knn_classifier(TrainSet, TestSet,'RCM', '4', 'LERM', k);
rcm_frobenius_accuracy = rcm_knn_classifier(TrainSet, TestSet,'RCM', '4', 'Frobenius', k);


%% show final results
fprintf('\n\n## Summary of results\n\n')

fprintf('##  Covariance Type: GRCM\n');
fprintf('##  GRCM: EV Accuracy = %5.2f\n', grcm_ev_accuracy);
fprintf('##  GRCM: EV2 Accuracy = %5.2f\n', grcm_ev2_accuracy);
fprintf('##  GRCM: Stein Accuracy = %5.2f\n', grcm_stein_accuracy);
fprintf('##  GRCM: AIRM Accuracy = %5.2f\n', grcm_airm_accuracy);
fprintf('##  GRCM: LERM Accuracy = %5.2f\n', grcm_lerm_accuracy);
fprintf('##  GRCM: Frobenius Accuracy = %5.2f\n\n', grcm_frobenius_accuracy);


fprintf('##  Covariance Type: RCM\n');
fprintf('##  RCM: EV Accuracy = %5.2f %%\n', rcm_ev_accuracy);
fprintf('##  RCM: EV2 Accuracy = %5.2f %%\n', rcm_ev2_accuracy);
fprintf('##  RCM: Stein Accuracy = %5.2f %%\n', rcm_stein_accuracy);
fprintf('##  RCM: AIRM Accuracy = %5.2f %%\n', rcm_airm_accuracy);
fprintf('##  RCM: LERM Accuracy = %5.2f %%\n', rcm_lerm_accuracy);
fprintf('##  RCM: Frobenius Accuracy = %5.2f %%\n', rcm_frobenius_accuracy);


