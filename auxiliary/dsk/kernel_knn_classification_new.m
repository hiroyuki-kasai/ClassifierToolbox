%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [accu,prediction] = kernel_knn_classification(test_kernel,train_label,K,test_label)
% this function performs classification with given kernels and labels
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input parameters:
% test_kernel: the test kernel
% train_label: the training label
% K: Ks of knn classifier
% test_label: the test label 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output parameters:
% accu: the classification accuracy
% prediction: the predicted label for the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jianjia Zhang, jz163@uowmail.edu.au Dec, 2014, all rights reserved
% For implementation details, please refer to: 
% "Learning Discriminative Stein Kernel for SPD Matrices and Its Applications." 
% arXiv preprint arXiv:1407.1974 (2014).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [accu,prediction] = kernel_knn_classification_new(test_kernel,train_label,K,test_label,options)
    ntest = size(test_kernel,2);
    if(size(test_kernel,1) ~= length(train_label))
        test_kernel = test_kernel';
        ntest = size(test_kernel,2);
    end
    test_label = test_label';
    train_label = train_label';
    nk = size(K,2);
    prediction = zeros(ntest,nk);
    for itest = 1:ntest
        cur_test = test_kernel(:,itest);
        [Y,I] = sort(cur_test,'descend');
        ttrain_label = train_label(I);
        for ik = 1:nk
            ktrain_label = ttrain_label(1:K(ik));

            ulabel = unique(ktrain_label);

            nlabel = size(ulabel,1);
            count = zeros(1,nlabel);
            for ilabel = 1:nlabel
                count(ilabel) = sum(ktrain_label==ulabel(ilabel));
            end
            [~,I] = max(count);
            prediction(itest,ik) = ulabel(I);
        end

       if options.verbose
            correct = (ulabel(I) == test_label(itest));
            fprintf('# Kernel kNN classifier: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', itest, ulabel(I), test_label(itest), correct);
       end

    end


    for ik = 1:nk
        accu(ik) = sum(prediction(:,ik) == test_label)/ntest;
    end
end
