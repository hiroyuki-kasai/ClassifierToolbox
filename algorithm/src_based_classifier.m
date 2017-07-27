function label = src_based_classifier(sparse_code, A, A_label, y, y_labal, classes, index, class_num, name, verbose)
% Classifier based on sparse representation classification (SRC) algorithm
%
% Inputs:
%       sparse_code         sparse code vector of of size nx1, where n is number of sets 
%       A                   dictionary of size dxn, where d is dimension and n is number of sets 
%       A_label             dictionary label of size 1xn, where n is number of sets
%       y                   observation vector of size dx1, where d is dimension 
%       y_label             label of observation
%       class_num           numner of classes
%       index               observation index
%       name                name of caller function
%       verbose             verbose mode flag
% Output:
%       label               classification label
%
% References:
%       J. Wright, A. Yang, A. Ganesh, S. Sastry, and Y. Ma, 
%       "Robust face recognition via sparse representation," 
%       IEEE Transaction on Pattern Analysis and Machine Intelligence, vol.31, no.2, pp.210-227, 2009.
%
%
% Created by H.Kasai on July 06, 2017


    % calculate sparse code
    %xp = l1_ls(TrainSet.X, y, lambda, 1e-3, 1); 
%     param.lambda = lambda;
%     param.lambda2 =  0; 
%     param.mode = 2;
%     xp = full(mexLasso(y, A, param));        

    % prepare residual array
    residuals = zeros(1, class_num);

    % calculate residual for each class
    for j = 1 : class_num
        idx = find(A_label == classes(j));
        %residuals(j) = norm(y - A(:,idx)*sparse_code(idx))/sum(sparse_code(idx).*sparse_code(idx));
        residuals(j) = norm(y - A(:,idx)*sparse_code(idx));
    end

    % calculate the predicted label with minimum residual
    [~, label] = min(residuals); 

    if verbose
        correct = (label == y_labal);
        fprintf('# %s: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', name, index, label, y_labal, correct);
    end 
    
end

