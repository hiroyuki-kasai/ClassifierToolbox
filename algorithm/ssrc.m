function accuracy = ssrc(TrainSet, TestSet, train_num, test_num, class_num, lambda, options)
% Superposed sparse representation based classification (SSRC) algorithm
%
% Inputs:
%       TrainSet            train sets of size dxn, where d is dimension and n is number of sets 
%       TestSet             test sets of size dxn, where d is dimension and n is number of sets
%       test_num            numner of test sets
%       class_num           numner of classes
%       lambda              regularization paramter
% Output:
%       accuracy            classification accurary
%
% References:
%       W. Deng, J. Hu, and J. Guo, 
%       "In defense of sparsity based face recognition,"
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR2013), 2013.
%
%
% Created by H.Kasai on July 11, 2017


    % extract options
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end
    
    if ~isfield(options, 'eigenface_dim')
        pca_dim = train_num;
    else
        pca_dim = options.eigenface_dim;
    end     


    % generate intra-class variant dictionary (base)
    classes = unique(TrainSet.y); 
    dim = size(TrainSet.X, 1);
    TrainSet.V = zeros(dim, train_num);
    for j = 1 : class_num
        idx = find(TrainSet.y == classes(j)); 

        data = TrainSet.X(:, idx);
        centroid = sum(data,2)/size(data,2);
        
        TrainSet.P(:, j) = centroid;                        % (Eq.9)

        % calculate logmap of centroid to each sample
        len = length(idx);
        for k = 1 : len
            diff = TrainSet.X(:, idx(k)) - centroid;
            TrainSet.V(:, idx(k)) = real(diff);             % (Eq.10)
        end

        if verbose
            fprintf('# Generating IntraVariDictionary for class %d\n', j);
        end
    end 
    

    % perform pca
    [disc_set, ~, ~]  =  Eigenface_f(TrainSet.X, pca_dim);
    % reduce dimension
    TrainSet.P = disc_set' * TrainSet.P;                    % (Eq.11)
    TrainSet.V = disc_set' * TrainSet.V;                    % (Eq.11)
    TestSet.X  = disc_set' * TestSet.X;
    

    % normalize data to l2-norm
    [TrainSet.P, ~] = data_normalization(TrainSet.P, TrainSet.y, 'std'); 
    [TrainSet.V, ~] = data_normalization(TrainSet.V, TrainSet.y, 'std'); 
    [TestSet.X, ~] = data_normalization(TestSet.X, TestSet.y, 'std');  
     
    
    % generate P and V set and its labels
    PV.X = [TrainSet.P TrainSet.V];
    PV.y = [classes, TrainSet.y];

    % prepare class array
    classes = unique(PV.y);
    
    % prepare predicted label array
    identity = zeros(1, test_num);
    
    for i = 1 : test_num

        y = TestSet.X(:, i);

        % calculate sparse code
        %xp = l1_ls(PV, y, lambda, 1e-3, 1); 
        param.lambda = lambda;
        param.lambda2 =  0; 
        param.mode = 2;
        alpha_beta = full(mexLasso(y, PV.X, param));        % (Eq.12)           

        % prepare residual array
        residuals = zeros(1, class_num);
        
        % calculate residual for each class
        for j = 1 : class_num
            non_idx = find(PV.y ~= classes(j));
            sc = alpha_beta;
            sc(non_idx) = 0;
            residuals(j) = norm(y - PV.X*sc)/sum(sc.*sc);   % (Eq.13) 
        end

        % calculate the predicted label with minimum residual
        [~, label] = min(residuals); 
        identity(i) = label;
        
        if verbose
            correct = (label == TestSet.y(1, i));
            fprintf('# SSRC: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', i, label, TestSet.y(1, i), correct);
        end           

    end

    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num/test_num;
end



