function accuracy = esrc(TrainSet, TestSet, train_num, test_num, class_num, lambda, options)
% Extended sparse representation classification (ESRC) algorithm
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
%       "Extended SRC: Undersampled face recognition via intraclass variant dictionary,"
%       IEEE Transation on Pattern Analysis Machine Intelligence, vol.34, no.9, pp.1864-1870, 2012.
%
%
% Created by H.Kasai on July 06, 2017


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


    % generate intra-class variation dictionary
    [TrainSet.X_sample_mean, ~] = data_normalization(TrainSet.X, TrainSet.y, 'sample_mean_per_class');     

    % generate eigenface
    if eigenface    
        [disc_set, ~, ~]  =  Eigenface_f(TrainSet.X, eigenface_dim);
        
        % project on subspace
        TrainSet.X  =  disc_set' * TrainSet.X;
        TestSet.X   =  disc_set' * TestSet.X;
        TrainSet.X_sample_mean = disc_set' * TrainSet.X_sample_mean;
    end

    % normalize data to l2-norm
    [TrainSet.X, ~] = data_normalization(TrainSet.X, TrainSet.y, 'std');   
    [TestSet.X, ~] = data_normalization(TestSet.X, TestSet.y, 'std');  
    [TrainSet.X_sample_mean, ~] = data_normalization(TrainSet.X_sample_mean, TrainSet.y, 'std');  
    
    % combine train set and intra-class variation dictionary
    combined_X = [TrainSet.X TrainSet.X_sample_mean];

    % prepare class array
    classes = unique(TrainSet.y);
    
    % prepare predicted label array
    identity = zeros(1, test_num);
    
    for i = 1 : test_num

        y = TestSet.X(:, i);

        % calculate sparse code
        %xp = l1_ls(combined_X, y, lambda, 1e-3, 1); 
        param.lambda = lambda;
        param.lambda2 =  0; 
        param.mode = 2;
        xp = full(mexLasso(y, combined_X, param));            

        % calcualte retraction error by variation dictionary 
        y = y - TrainSet.X_sample_mean * xp(train_num+1:end);

        % prepare residual array
        residuals = zeros(1, class_num);
        
        % calculate residual for each class
        for j = 1 : class_num
            idx = find(TrainSet.y == classes(j));
            residuals(j) = norm(y-TrainSet.X(:,idx)*xp(idx))/sum(xp(idx).*xp(idx));
            %residuals(j) = norm(y - TrainSet.X(:,idx)*xp(idx));
        end

        % calculate the predicted label with minimum residual
        [~, label] = min(residuals); 
        identity(i) = label;
        
        if verbose
            correct = (label == TestSet.y(1, i));
            fprintf('# ESRC: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', i, label, TestSet.y(1, i), correct);
        end           

    end

    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num/test_num;
end



