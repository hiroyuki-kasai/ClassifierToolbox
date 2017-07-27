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
    
    if ~isfield(options, 'pca')
        pca = true;
    else
        pca = options.pca;
    end     
    
    if ~isfield(options, 'pca_dim')
        pca_dim = train_num;
    else
        pca_dim = options.pca_dim;
    end     


    % generate intra-class variant dictionary (base)
    classes = unique(TrainSet.y);
    if ~isfield(TrainSet, 'P') && ~isfield(TrainSet, 'V')
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
    end
    

    % perform pca
    if pca
        [disc_set, ~, ~]  =  Eigenface_f(TrainSet.X, pca_dim);
        % reduce dimension
        TrainSet.P = disc_set' * TrainSet.P;                    % (Eq.11)
        TrainSet.V = disc_set' * TrainSet.V;                    % (Eq.11)
        TestSet.X  = disc_set' * TestSet.X;
    end
    

    % normalize data to l2-norm
    [TrainSet.P, ~] = data_normalization(TrainSet.P, TrainSet.y, 'std'); 
    [TrainSet.V, ~] = data_normalization(TrainSet.V, TrainSet.y, 'std'); 
    [TestSet.X, ~] = data_normalization(TestSet.X, TestSet.y, 'std');  
     
    
    % generate P and V set and its labels
    PV.X = [TrainSet.P TrainSet.V];
    %PV.y = [classes, TrainSet.y];
    PV.y = classes;

    % prepare class array
    classes = unique(PV.y);
    
    % prepare predicted label array
    identity = zeros(1, test_num);
    
    for i = 1 : test_num

        y = TestSet.X(:, i);
        
        % calculate sparse code
        if 0
            %tau = max(1e-4*max(abs(TrainSet.X'*y)),sigma*sqrt(log(train_num)));
            tau = lambda;

            in = [];   
            in.tau = tau;
            delx_mode = 'qr'; % mil or qr
            in.delx_mode = delx_mode;
            in.debias = 0;
            in.verbose = 0;
            in.plots = 0;

            out = l1homotopy(PV.X, y, in);
            alpha_beta = out.x_out;
            
        elseif 1
            
            P = inv(PV.X'*PV.X+0.001*eye(size(PV.X,2)))*PV.X';
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
            [alpha_beta, iterationCount] = SolveHomotopy(PV.X, y, ...
                            'maxIteration', maxIteration,...
                            'isNonnegative', isNonnegative, ...
                            'stoppingCriterion', stoppingCriterion, ...
                            'groundtruth', x0, ...
                            'lambda', lambda, ...
                            'tolerance', tolerance);              
        elseif 0
            alpha_beta = l1_ls(PV.X, y, 1e-3); 
        else
        
            param.lambda = lambda;
            param.lambda2 =  0; 
            param.mode = 0;
            alpha_beta = full(mexLasso(y, PV.X, param));         % (Eq.12)     
        end        


        % prepare residual array
        residuals = zeros(1, class_num);
        
        % calculate residual for each class
        for j = 1 : class_num
            non_idx = find(PV.y ~= classes(j));
            sc = alpha_beta;
            sc(non_idx) = 0;
            %residuals(j) = norm(y - PV.X*sc)/sum(sc.*sc);   % (Eq.13) 
            residuals(j) = norm(y - PV.X*sc);   % (Eq.13) 
        end

        % calculate the predicted label with minimum residual
        [~, label] = min(residuals); 
        identity(i) = label;
        
        if verbose
            correct = (label == TestSet.y(1, i));
            fprintf('# SSRC: test:%04d, predict class: %03d --> ground truth :%03d (%d)\n', i, label, TestSet.y(1, i), correct);
        end           

    end

    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num/test_num;
end



