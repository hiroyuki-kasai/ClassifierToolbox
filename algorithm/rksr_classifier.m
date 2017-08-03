function accuracy = rksr_classifier(TrainSet, TestSet, options)
% Riemannian kernelized sparse representation classification (R-KSRC) algorithm
%
% Inputs:
%       TrainSet            train sets of size dxn, where d is dimension and n is number of sets 
%       TestSet             test sets of size dxn, where d is dimension and n is number of sets
%       options             options
% Output:
%       accuracy            classification accurary
%
% References:
%       M. Harandi, R. Hartley, B. Lovell and C. Sanderson, 
%       "Sparse coding on symmetric positive definite manifolds using bregman divergences," 
%       IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2016.
%
%       M. Harandi, C. Sanderson, R. Hartley and B. Lovell, 
%       "Sparse coding and dictionary learning for symmetric positive definite matrices: a kernel approach," 
%       European Conference on Computer Vision (ECCV), 2012.
%
% Originally created by Mehrtash Harandi (mehrtash.harandi at gmail dot com).
% Modified by H. Kasai on July 10, 2017.
    
    
    % retrieve dimension of the SPD matrices
    dim = size(TrainSet.X_cov, 1);
    test_num = length(TestSet.y);
    class_num = length(unique(TestSet.y));
    
    % calculate eigen decomposition
    train_decomp = Decomposite_eig_new(TrainSet);
    test_decomp = Decomposite_eig_new(TestSet);
       
    % compute the Stein divergence with the obtained adjustment parameter optimal_alpha
    S_test          = EigComp2SD_power_new(train_decomp, test_decomp, ones(1,dim)); 
    S_train         = EigComp2SD_power_new(train_decomp, train_decomp, ones(1,dim));
    train_kernel    = exp(-1 * options.theta * S_train);    
    test_kernel     = exp(-1 * options.theta * S_test);

    % normalize dictionary
    [KD, ~] = data_normalization(train_kernel, [], 'std');   
    [KX, ~] = data_normalization(test_kernel, [], 'std');   


    [KD_U, KD_D, ~] = svd(KD);    
    A = diag(sqrt(diag(KD_D))) * KD_U';
    D_Inv = KD_U * diag(1./sqrt(diag(KD_D)));
    KX = D_Inv' * KX;
    
    % perform lasso
    param.lambda = options.lambda;
    param.lambda2 =  0; 
    param.mode = 2;
    scX = full(mexLasso(KX, A, param));
 

    % prepare class array
    classes = unique(TrainSet.y);    
    % prepare predicted label array
    identity = zeros(1, test_num);
    
    for i = 1 : test_num
        % prepare residual array
        residuals = zeros(1, class_num);
        
        % calculate residual for each class
        for j = 1 : class_num
            idx = find(TrainSet.y == classes(j));
            %residuals(j) = norm(KX(:, i) - A(:,idx)*scX(idx, i))/sum(scX(idx, i) .* scX(idx, i));
            residuals(j) = norm(KX(:, i) - A(:,idx)*scX(idx, i));
            
            if strcmp(options.mode, 'src')
                residuals(j) = norm(KX(:, i) - A(:,idx)*scX(idx, i));                
            elseif strcmp(options.mode, 'ip_linear')
                scX_i = scX(idx, i);
                %residuals(j) = sum(abs(scX_i));
                residuals(j) = sum(scX_i);
            elseif strcmp(options.mode, 'ip_max')
                scX_i = scX(idx, i);
                residuals(j) = max(abs(scX_i));
            end
        end

        % calculate the predicted label
        [~, label] = min(residuals); 
        if strcmp(options.mode, 'src')
            [~, label] = min(residuals);                
        elseif strcmp(options.mode, 'ip_linear') || strcmp(options.mode, 'ip_max')
            [~, label] = max(residuals); 
        end
        identity(i) = label;
        
        if options.verbose
            correct = (label == TestSet.y(1, i));
            fprintf('# R-KSR (with %s metric): test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', options.mode, i, label, TestSet.y(1, i), correct);
        end           
    end

    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num/test_num;
end


