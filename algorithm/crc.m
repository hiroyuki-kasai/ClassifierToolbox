function accuracy = crc(TrainSet, TestSet, train_num, test_num, class_num, lambda, options)
% Collaborative representation based classification (CRC) algorithm
%
% Inputs:
%       TrainSet            train sets of size dxn, where d is dimension and n is number of sets 
%       TestSet             test sets of size dxn, where d is dimension and n is number of sets
%       train_num           numner of train sets
%       test_num            numner of test sets
%       class_num           numner of classes
%       lambda              regularization paramter
% Output:
%       accuracy            classification accurary
%
% References:
%       Lei Zhanga, Meng Yanga, and Xiangchu Feng
%       "Sparse Representation or Collaborative Representation: Which Helps Face Recognition?,"
%       Proceedings of the 2011 International Conference on Computer Vision (ICCV'11), pp. 471-478, 2011.
%
%
% Created by H.Kasai on July 04, 2017
%
% Note that this code partially refers the codes written by Meng Yang @ COMP HK-PolyU. 


    % extract options
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end
    
    if ~isfield(options, 'eigenface')
        eigenface = true;
    else
        eigenface = options.eigenface;
    end    
    
    if ~isfield(options, 'eigenface_dim')
        eigenface_dim = train_num;
    else
        eigenface_dim = options.eigenface_dim;
    end  


    % calculate eigenface
    if eigenface
        [disc_set, ~, ~] = Eigenface_f(TrainSet.X, eigenface_dim);
        TrainSet.X_red  =  disc_set' * TrainSet.X;
        TestSet.X_red  =  disc_set' * TestSet.X;
    else
        TrainSet.X_red  =  TrainSet.X;
        TestSet.X_red  =  TestSet.X;        
    end
    

    % normalize data to l2-norm
    [TrainSet_normalized.X, TrainSet_normalized.y] = data_normalization(TrainSet.X, TrainSet.y, 'std');   
    [TestSet_normalized.X, TestSet_normalized.y] = data_normalization(TestSet.X, TestSet.y, 'std');              


    % calculate projection matrix 
    %P = inv(TrainSet_normalized.X' * TrainSet_normalized.X + lambda * eye(size(TrainSet_normalized.X,2))) * TrainSet_normalized.X';
    P = (TrainSet_normalized.X' * TrainSet_normalized.X + lambda * eye(size(TrainSet_normalized.X,2))) \ TrainSet_normalized.X';

    identity = zeros(1, test_num);
    for i = 1 : test_num
        
        y = TestSet_normalized.X(:,i);
        
        % CRC RLS classification function
        rho_hat =  P * y;
        err_array = zeros(1, class_num);
        for j = 1 : class_num
            rho_hat_class_j = rho_hat(TrainSet_normalized.y==j);
            X_class_j =  TrainSet_normalized.X(:,TrainSet_normalized.y==j);
            err_array(j) = norm(y - X_class_j*rho_hat_class_j) / sum(rho_hat_class_j.*rho_hat_class_j); % Eq.(10)
        end

        [~, label] = min(err_array);
        identity(i) = label;
        
        if verbose
            correct = (label == TestSet.y(1, i));
            fprintf('# CRC: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', i, label, TestSet.y(1, i), correct);            
        end     
    end
    
    
    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num/test_num; 
end

