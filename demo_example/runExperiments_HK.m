% =========================================================================
%   Michael Iliadis, Leonidas Spinoulas, Albert Berahas, Haohong Wang and
%   Aggelos K. Katsaggelos
%
%
% Written by Michael Iliadis @ NU-IVPL
% March, 2014.
% =========================================================================

%clearvars;
clear;
clc;
close all;

%% PARAMETERS
method = 'src+rls'; % 'cr-rls', 'esrc', 'src+rls'
dataset = 'yale'; % 'yale' or 'ar'

%% START EXPERIMENTAL ROUNDS

fr.dataset = dataset;
fr.method = method;

original = 1;

if original
    %% Dataset selection
    data = datasetSelection(fr.dataset,	1);

    %% Initializations
    [train,test,accuracy,groundtruth_train,groundtruth_test] = ...
        initialize( data);
    
    if 1
        TrainSet.X = train;
        TrainSet.y = groundtruth_train;
        TestSet.X = test;
        TestSet.y = groundtruth_test;

        max_class_num = 10;
        max_samples = 10;
        [TrainSet, TestSet, train_num, test_num, class_num] = ... 
            reduce_dataset(TrainSet, TestSet, max_class_num, max_samples);

        train = TrainSet.X;
        groundtruth_train = TrainSet.y;
        test = TestSet.X;
        groundtruth_test = TestSet.y;
    end
    

    %% Normalize data
    store_normz = normalize_data(groundtruth_train,train,test);
    
    %% Create method's parameters
    pars = parSelection(fr.method, groundtruth_train, store_normz);       
else
    %% load dataset
    %load('../dataset/ORL_Face_img.mat');
    %load('../dataset/Brodatz_texture_img_small_set.mat');
    %load('../dataset/Cropped_YaleB_small_img_24x20.mat'); % does not work. Why???
    %load('../dataset/AR_Face_img_27x20_from_LKDL.mat'); 
    load('../dataset/AR_Face_img_60x43_from_CRC.mat'); 
    % 
    max_class_num = 20;
    max_samples = 6;
    [TrainSet, TestSet, train_num, test_num, class_num] = reduce_dataset(TrainSet, TestSet, max_class_num, max_samples);

%     TrainSetOrg = TrainSet;
%     TestSetOrg = TestSet;
% 
%     %% Normalize data
%     store_normz = normalize_data(TrainSet.y, TrainSet.X, TestSet.X);
%     
%     TrainSet.X = store_normz.trainData;
%     TestSet.X = store_normz.testData;
%     
%     %% Create method's parameters
%     
%     pars = parSelection(fr.method, TrainSet.y, store_normz);  
end

if 1
%% RUN Face Recognition
%parfor ii=1:size(store_normz.testData,2)
for ii=1:size(store_normz.testData,2)
    warning('off','all');

    test = store_normz.testData(:,ii);
    [label,residuals] = runFR(method, pars, test);
    
    %label
    
    % Create accuracy
    if original
        correct_label = groundtruth_test(ii);
    else
        correct_label = TestSet.y(ii);
    end

    if label==correct_label
        accuracy(ii) = 1;
        fprintf('## %d right!\n', ii);
    else
        accuracy(ii) = 0;
        %fprintf('%d wrong! %d place: %d\n', ii,groundtruth_test(ii),find(groundtruth_test(ii)==residuals));
        fprintf('-- %d wrong!\n', ii);
    end
end


if original 
    test_num = size(store_normz.testData,2);
else
    test_num = length(TestSet.y);
end

accuracy_ratio = nnz(accuracy)/test_num;
fprintf('Accuracy = %5.5f\n', accuracy_ratio);
end

return;

%TrainSet = TrainSetOrg;
%TestSet = TestSetOrg;


% normalization_method = 'sample_mean_per_class';
% [TrainSet.X_sample_mean, ~, ~, ~] = ...
%             dataset_normalization(normalization_method, TrainSet.X, TrainSet.y, TestSet.X, TestSet.y);
        
% [TrainSet.X_sample_mean, ~] = data_normalization(TrainSet.X, TrainSet.y, 'sample_mean_per_class');     
% 
% 
% % go to eigenfaces
% [disc_set, disc_value, mean_img]  =  eigenFace(TrainSet.X, 15);
% %[TrainSet.X, TestSet.X, V_new, P_new, mean_img] = goEigenface(disc_set, disc_value, mean_img, TrainSet.X,TestSet.X, V_new, P_new);
% 
% TrainSet.X  =  disc_set' * TrainSet.X;
% TestSet.X   =  disc_set' * TestSet.X;
% TrainSet.X_sample_mean = disc_set' * TrainSet.X_sample_mean;
% 
% 
% 
% % normalize data to l2-norm
% % TrainSet.X  =  normc(TrainSet.X);
% % TestSet.X  =  normc(TestSet.X);
% % TrainSet.X_sample_mean = normc(TrainSet.X_sample_mean);
% [TrainSet.X, ~] = data_normalization(TrainSet.X, TrainSet.y, 'std');   
% [TestSet.X, ~] = data_normalization(TestSet.X, TestSet.y, 'std');  
% [TrainSet.X_sample_mean, ~] = data_normalization(TrainSet.X_sample_mean, TrainSet.y, 'std');  



lambda = 0.001;
%IntraVarDic = TrainSet.X_sample_mean;
options.verbose = true;
options.eigenface = true;
options.eigenface_dim = 15;
accuracy_esrc = esrc(TrainSet, TestSet, train_num, test_num, class_num, lambda, options);
fprintf('ESRC Accuracy = %5.5f\n', accuracy_esrc);


% lambda = 0.001;
% options.verbose = true;
% accuracy_crrls = crrls(TrainSet, TestSet, train_num, test_num, class_num, lambda, options);
% fprintf('CR-RLS Accuracy = %5.5f\n', accuracy_crrls);

%% CRC
dimension = 15; % This should not be greater than (max_class_num*max_samples).
lambda = 0.001;
options.verbose = false;
options.eigenface = true;
accuracy_crc = crc(TrainSet, TestSet, test_num, class_num, dimension, lambda, options);
fprintf('# CRC: Accuracy = %5.5f\n', accuracy_crc);

return;


%% PRINT FINAL RECOGNITION RATE
meanAcc = mean(acc);
stdAcc = std(acc);
fprintf('Recognition rate: %.2f and Std: %.2f\n', meanAcc*100,stdAcc*100);