function accuracy = lcdrc(TrainSet, TestSet, train_num, test_num, class_num, reduce_dimension, options)
% Linear collaborative discriminant regression classificatoin (LCDRC) algorithm
%
% Inputs:
%       TrainSet            train sets of size dxn, where d is dimension and n is number of sets 
%       TestSet             test sets of size dxn, where d is dimension and n is number of sets
%       test_num            numner of test sets
%       class_num           numner of classes
%       options             options
% Output:
%       accuracy            classification accurary
%
% References:
%       X. Qu, S. Kim, R. Cui and H. J. Kim,
%       "Linear collaborative discriminant regression classification for face recognition,"
%       J. Visual Communication Image Represetation, vol.31, pp. 312-319, 2015.
%
%
% Created by H.Kasai on July 04, 2017

    % extract options
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end
    

    if verbose
        fprintf('# LCDRC: calculate projection matrix U ...');
    end
    Eb = 0;
    Ew = 0;
    for j = 1 : train_num

        j_class = TrainSet.y(1, j);

        % calcuate Ew
        intra_class_index = find(TrainSet.y == j_class);
        intra_class_index(intra_class_index == j) = [];
        X_intra_wo_j = TrainSet.X(:, intra_class_index);                                    % X^{intra}
        alpha_intra = X_intra_wo_j * pinv(X_intra_wo_j' * X_intra_wo_j) * X_intra_wo_j';    % alpha^{intra} in Eq.(2)
        error_intra = TrainSet.X(:,j) - alpha_intra * TrainSet.X(:,j);
        Ew = Ew + error_intra * error_intra';

        % calcuate Eb
        inter_class_index = find(TrainSet.y ~= j_class);
        X_inter = TrainSet.X(:, inter_class_index);%                                        % X^{iner}
        alpha_inter= X_inter * pinv(X_inter' * X_inter) * X_inter';                         % alpha^{inter} in Eq.(2)
        error_inter = TrainSet.X(:,j) - alpha_inter * TrainSet.X(:,j); 
        Eb = Eb + error_inter * error_inter';
    end

    % calc
    Ew = Ew/train_num;
    Eb = Eb/train_num;
    lambda = 0.001;
    Ew = Ew + lambda*eye(size(Ew,2));

    [V, ~] = eig(Eb, Ew);
    for i = 1:reduce_dimension
        U(:,i) = V(:,size(V,2)+1-i);                                                        % Eq.(13)
    end
    if verbose
        fprintf('done\n');
    end    

    
    % reduce dimention
    TrainSet.X_red  = U' * TrainSet.X;
    TestSet.X_red   = U' * TestSet.X;        


    % prepare projection matrix (hat matrix)
    H = cell(1, class_num);
    for i = 1 : class_num
        class_index = find(TrainSet.y == i);

        X_i = TrainSet.X_red(:, class_index);           % Eq.(1)
        alpha_i = pinv(X_i' * X_i) *  X_i';                  % Eq.(2) exept y (beta_i = inv(X_i' * X_i) * X_i')
        H{i} = X_i * alpha_i;                           % Eq.(3) exept y

        if verbose
            fprintf('# LCDRC: calc Hi for class : %03d/%03d (samples: %03d)\n', i, class_num, length(class_index));
        end
    end
    
    
    % prepare predicted label array
    identity = zeros(1, test_num); 

    % predict the class
    for j = 1 : test_num

        dis = zeros(1, class_num);
        y = TestSet.X_red(:, j);
        for i = 1 : class_num
            y_pred = H{i} * y;
            dis(1, i) = norm(y - y_pred);               % Eq.(5)
        end
        [~, label] = min(dis);                          % Eq.(6)
        identity(j) = label;        

        if verbose
            correct = (label == TestSet.y(1, j));
            fprintf('# LCDRC: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', j, label, TestSet.y(1, j), correct);
        end
    end
    
    
    % calculate accuracy
    correct_num = sum(identity == TestSet.y);    
    accuracy = correct_num / test_num;
end

