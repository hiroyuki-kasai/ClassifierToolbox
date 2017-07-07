function accuracy = rpca_src(TrainSet, TestSet, train_num, test_num, class_num, lambda, options)
% Robust PCA-based sparse representation classification (RobustPCA-SRC) algorithm
%
% Inputs:
%       TrainSet            train sets of size dxn, where d is dimension and n is number of sets 
%       train_num           numner of train sets
%       class_num           numner of classes
%       lambda              regularization paramter
% Output:
%       accuracy            classification accurary
%
%
% Created by H.Kasai on July 07, 2017
    

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
    
    if ~isfield(options, 'solver_max_iter')
        solver_max_iter = 100;
    else
        solver_max_iter = options.solver_max_iter;
    end    
  
    
    % define problem definitions
    solver_lambda = 1/sqrt(max(size(TrainSet.X)));
    solver_lambda = solver_lambda/3;
    mask = logical(zeros(size(TrainSet.X)));
    problem = robust_pca(TrainSet.X, mask, solver_lambda);
    problem = robust_pca(TrainSet.X, mask, solver_lambda);

    % perform robust PCA
    solver_options.max_iter = solver_max_iter;
    solver_options.verbose = verbose;  
    solver_options.mu = 10*solver_lambda/3;
    [w, ~] = admm_robust_pca(problem, solver_options);
    TrainSet.X = w.L;
    
    problem = robust_pca(TestSet.X, mask, solver_lambda);
    [w, ~] = admm_robust_pca(problem, solver_options);
    TestSet.X = w.L;    
    
    % generate eigenface
    if eigenface    
        [disc_set, ~, ~] = Eigenface_f(TrainSet.X, eigenface_dim);
        
        % project on subspace
        TrainSet.X  =  disc_set' * TrainSet.X;
        TestSet.X   =  disc_set' * TestSet.X;
    end

    % normalize data to l2-norm
    [TrainSet.X, ~] = data_normalization(TrainSet.X, TrainSet.y, 'std');   
    [TestSet.X, ~] = data_normalization(TestSet.X, TestSet.y, 'std');  
    
    
    % perform a standard SRC
    if 1
        % use src function
        options.eigenface = false;
        accuracy = src(TrainSet, TestSet, train_num, test_num, class_num, lambda, options);    
    else
    
        % prepare class array
        classes = unique(TrainSet.y);

        % prepare predicted label array
        identity = zeros(1, test_num);

        for i = 1 : test_num

            y = TestSet.X(:, i);

            % calculate sparse code
            xp = l1_ls(TrainSet.X, y, lambda, 1e-3, 1); 

            % prepare residual array
            residuals = zeros(1, class_num);

            % calculate residual for each class
            for j = 1 : class_num
                idx = find(TrainSet.y == classes(j));
                residuals(j) = norm(y-TrainSet.X(:,idx)*xp(idx))/sum(xp(idx).*xp(idx));
                %residuals(j) = norm(y - TrainSet.X(:,idx)*xp(idx));
            end

            % calculate the predicted label with minimum residual
            [dis, label] = min(residuals); 
            identity(i) = label;

            if verbose
                correct = (label == TestSet.y(1, i));
                fprintf('# RobustPCA-SRC: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', i, label, TestSet.y(1, i), correct);
            end           

        end
    

        % calculate accuracy
        correct_num = sum(identity == TestSet.y);
        accuracy = correct_num/test_num;
    end
end


