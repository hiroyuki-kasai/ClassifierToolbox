function [accuracy, W, b] = lsr(TrainSet, TestSet, train_num, test_num, class_num, lambda, options)
% Least squares regression (LSR) algorithm
%
% Inputs:
%       TrainSet            train sets of size dxn, where d is dimension and n is number of sets 
%       train_num           numner of train sets
%       class_num           numner of classes
%       lambda              regularization paramter
% Output:
%       accuracy            classification accurary
%       W                   projection matrix
%       b                   offset
%
%
% Created by H.Kasai on July 04, 2017
% Modified by H.Kasai on July 06, 2017
    

    % extract options
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end
    
    dim = size(TrainSet.X, 1);
    
    % convert lavel vector to matrix representation
    TrainSet.y_mat = convert_labelvec_to_mat(TrainSet.y, train_num, class_num);
    
    % subtract mean vales of dimentional direction
    [TrainSet.X, ~] = data_normalization(TrainSet.X, TrainSet.y, 'dimension_mean');   
    [TestSet.X, ~] = data_normalization(TestSet.X, TestSet.y, 'dimension_mean');      

    % calculate projection matrix and b
    W = (TrainSet.X * TrainSet.X' + lambda * eye(dim)) \ TrainSet.X * TrainSet.y_mat';   
    b = TrainSet.y_mat - W' * TrainSet.X;
    b = mean(b, 2);
    
    % project test set onto subspace
    Y_pred = TestSet.X' * W;
    
    identity = zeros(1, test_num);
    for i = 1 : test_num
        [~, label] = max(Y_pred(i, :));
        identity(i) = label;
        
        if verbose
            correct = (label == TestSet.y(1, i));
            fprintf('# LSR: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', i, label, TestSet.y(i), correct);
        end        
    end
    
    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num/test_num; 
end


