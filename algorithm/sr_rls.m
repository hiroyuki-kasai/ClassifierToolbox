function accuracy = sr_rls(TrainSet, TestSet, train_num, test_num, class_num, lambda_l1, lambda_l2, options)
% Sparse representation and least squares-based classification (SRC-RLS) algorithm
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
%       M. Iliadis, L. Spinoulas, A. S. Berahas, H. Wang, and A. K. Katsaggelos, 
%       "Sparse representation and least squares-based classification in face recognition,"
%       Proceedings of the 22nd European Signal Processing Conference (EUSIPCO), 2014.
%
%
% Created by H.Kasai on July 11, 2017


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
        pca_dim = train_num;
    else
        pca_dim = options.eigenface_dim;
    end     


    % generate intra-class variant dictionary (base): P and V
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
            fprintf('# Generating P and V for class %d\n', j);
        end
    end 
       

    if eigenface  
        [disc_set, ~, ~]  =  Eigenface_f(TrainSet.X, pca_dim);
        % reduce dimension
        TrainSet.P = disc_set' * TrainSet.P;
        TrainSet.V = disc_set' * TrainSet.V;
        TrainSet.X = disc_set' * TrainSet.X;
        TestSet.X  = disc_set' * TestSet.X;
    end
    

    % normalize data to l2-norm
    [TrainSet.P, ~] = data_normalization(TrainSet.P, TrainSet.y, 'std'); 
    [TrainSet.V, ~] = data_normalization(TrainSet.V, TrainSet.y, 'std');
    [TrainSet.X, ~] = data_normalization(TrainSet.X, TrainSet.y, 'std');
    [TestSet.X, ~] = data_normalization(TestSet.X, TestSet.y, 'std');  
    
    
    % generate X and V set and its labels
    XV.X = [TrainSet.X TrainSet.V];
    XV.y = [TrainSet.y, TrainSet.y];

    % prepare class array
    classes = unique(XV.y);
    
    % prepare predicted label array
    identity = zeros(1, test_num);
     
    for i = 1 : test_num

        y = TestSet.X(:, i);

        % calculate sparse code
        %xp = l1_ls(PV, y, lambda, 1e-3, 1); 
        param.lambda = lambda_l1;
        param.lambda2 =  0; 
        param.mode = 2;
        alpha = full(mexLasso(y, XV.X, param));        % (Eq.12) 
        
        if nnz(alpha) > 0
        
            % chose active classes: (Eq.7)
            active_class_idx = [];
            cnt = 1;
            for j = 1 : class_num
                idx = TrainSet.y == classes(j);
                if find(alpha(idx)>0)
                %if find(abs(alpha(idx))>0)
                    active_class_idx(cnt) = j;
                    cnt = cnt+1;
                end
            end
            
            if ~isempty(active_class_idx)

                % generate new dictionary: tilde{T} : (Eq.8)
                new_dic = [];
                for aa = 1 : length(active_class_idx)
                    idx_columns = find(TrainSet.y == classes(active_class_idx(aa)));    
                    get_train = TrainSet.X(:,idx_columns(1):max(idx_columns));    
                    new_dic = [new_dic get_train];    
                end


                % solve regularized least squares (RLS) step: (Eq.9) and (Eq.10)
                [R, p] = chol(new_dic'*new_dic + eye(size(new_dic'*new_dic,2)).*lambda_l2);
                alpha_tmp = R \ (R' \ (new_dic'*y));             


                % generate new sparse coefficients (tilde{f}_i)
                new_alpha = zeros(train_num*2, 1);
                max_col = 0;
                for j = 1 : length(active_class_idx)
                    idx_columns = find(TrainSet.y == classes(active_class_idx(j)));
                    len = length(idx_columns);
                    new_alpha(idx_columns(1):max(idx_columns)) = alpha_tmp(max_col+1:max_col+len);
                    max_col = max_col+len;
                end        

                % prepare residual array
                residuals = zeros(1, class_num);

                % calculate residual for each class
                for j = 1 : class_num
                    idx = find(TrainSet.y == classes(j));
                    residuals(j) = norm(y - XV.X(:,idx) * new_alpha(idx),2);
                end

                % calculate the predicted label with minimum residual
                [~, label] = min(residuals); 
                identity(i) = label;
                
            else
                label = -1;
                identity(i) = label;                
            end
        else
            label = -1;
            identity(i) = label;
        end

        if verbose
            correct = (label == TestSet.y(1, i));
            fprintf('# SRC-RLS: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', i, label, TestSet.y(1, i), correct);
        end           

    end

    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num/test_num;
end



