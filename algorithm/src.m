function accuracy = src(TrainSet, TestSet, train_num, test_num, class_num, lambda, options)
% Sparse representation classification (SRC) algorithm
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
%       J. Wright, A. Yang, A. Ganesh, S. Sastry, and Y. Ma, 
%       "Robust face recognition via sparse representation," 
%       IEEE Transaction on Pattern Analysis and Machine Intelligence, vol.31, no.2, pp.210-227, 2009.
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
    
    % prepare class array
    classes = unique(TrainSet.y);
    
    % prepare predicted label array
    identity = zeros(1, test_num);
    
    for i = 1 : test_num

        y = TestSet.X(:, i);

        
        if 0
            % calculate sparse code
            %tau = max(1e-4*max(abs(TrainSet.X'*y)),sigma*sqrt(log(train_num)));
            tau = lambda;

            in = [];   
            in.tau = tau;
            delx_mode = 'mil'; % mil or qr
            in.delx_mode = delx_mode;
            in.debias = 0;
            in.verbose = 0;
            in.plots = 0;

            out = l1homotopy(TrainSet.X, y, in);
            xp = out.x_out;
        elseif 1
            
            P = inv(TrainSet.X'*TrainSet.X+0.001*eye(size(TrainSet.X,2)))*TrainSet.X';
            x0 = P*y;
            
            maxIteration = 5000;
            isNonnegative = false;
            lambda = 1e-2; %5e-3;
            tolerance = 0.05;
            STOPPING_GROUND_TRUTH = -1;
            STOPPING_DUALITY_GAP = 1;
            STOPPING_SPARSE_SUPPORT = 2;
            STOPPING_OBJECTIVE_VALUE = 3;
            STOPPING_SUBGRADIENT = 4;
            stoppingCriterion = STOPPING_GROUND_TRUTH;
            [xp, iterationCount] = SolveHomotopy(TrainSet.X, y, ...
                            'maxIteration', maxIteration,...
                            'isNonnegative', isNonnegative, ...
                            'stoppingCriterion', stoppingCriterion, ...
                            'groundtruth', x0, ...
                            'lambda', lambda, ...
                            'tolerance', tolerance);  
                        
                        
        elseif 0
            xp = l1_ls(TrainSet.X, y, 1e-3); 
        else
        
            param.lambda = lambda;
            param.lambda2 =  0; 
            param.mode = 0;
            xp = full(mexLasso(y, TrainSet.X, param));   
        end

        % prepare residual array
        residuals = zeros(1, class_num);
        
        % calculate residual for each class
        for j = 1 : class_num
            idx = find(TrainSet.y == classes(j));
            %residuals(j) = norm(y-TrainSet.X(:,idx)*xp(idx))/sum(xp(idx).*xp(idx));
            residuals(j) = norm(y - TrainSet.X(:,idx)*xp(idx));
        end

        % calculate the predicted label with minimum residual
        [~, label] = min(residuals); 
        identity(i) = label;
        
        if verbose
            correct = (label == TestSet.y(1, i));
            fprintf('# SRC: test:%04d, predict class: %03d --> ground truth :%03d (%d)\n', i, label, TestSet.y(1, i), correct);
        end 
        
        %identity(i) = src_based_classifier(xp, TrainSet.X, TrainSet.y, TestSet.X(:, i), TestSet.y(1, i), classes, i, class_num, 'SRC', verbose);        

    end

    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num/test_num;
end



