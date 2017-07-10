function accuracy = sdr_slr(TrainSet, TestSet, train_num, test_num, class_num, options)
% Sparse- and dense-hybrid representation (SDR) and supervised low-rank (SLR) dicrtionary decomposition (SDR-SLR) algorithm
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
%       X. Jiang, and J. Lai, 
%       "Sparse and dense hybrid representation via dictionary decomposition for face recognition," 
%       IEEE Transaction on Pattern Analysis and Machine Intelligence, vol.37, no.5, pp.1067-1079, 2015.
%
% Originally created by X. Jiang and J. Lai ({jlai1,exdjiang}@ntu.edu.sg).
% Modified by H.Kasai on July 10, 2017


    % extract options
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end
    
    
    % slr procedure
    [A, B, X, E] = slr_iteration(TrainSet.X, TrainSet.y, options.slr_lambda, options.slr_tau, options.slr_eta, options.slr_delta, 1e-4, 1000, 4);


    % prepare predicted label array
    identity = zeros(1, test_num);

    for i = 1 : test_num

        % get the ith testing image
        y = TestSet.X(:,i);

        % sdr procedure
        [a, x, e] = sdr_ialm(A, B, y, options.sdr_beta, options.sdr_gamma, 5e-3, 1000);

        % recovered clean face
        dny = y - B*x - e;

        % class residual
        d2 = zeros(1, class_num);
        for k = 1:class_num
            residual = dny - A(:,k==TrainSet.y) * a(k==TrainSet.y);
            d2(k) = residual' * residual;
        end
        
        % classification result
        [~, label] = min(d2);
        identity(i) = label;
        
        if verbose
            correct = (identity(i) == TestSet.y(1, i));
            fprintf('# SDR-SLR: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', i, label, TestSet.y(1, i), correct);
        end          
    end
    
    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num/test_num;
end



