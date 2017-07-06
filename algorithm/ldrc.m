function accuracy = ldrc(TrainSet, TestSet, train_num, test_num, class_num, reduce_dimension, options)
% Linear discriminant regression classificatoin (LDRC) algorithm
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
%       S.-M. Huang and J.-F. Yang,
%       "Linear discriminant regression classification for face recognition,"
%       IEEE Signal Processing Letters, vol.20, no.1, pp.91-94, 2013.
%
%
% Created by H.Kasai on July 05, 2017


    % extract options
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end


    if verbose
        fprintf('# LDRC: calculate projection matrix U ...');
    end
    Eb = 0;
    Ew = 0;
    for j = 1 : train_num

        j_class = TrainSet.y(1, j);

        % calcuate Ew
        intra_class_index = find(TrainSet.y == j_class);
        intra_class_index(intra_class_index == j) = [];
        X_intra_wo_j = TrainSet.X(:, intra_class_index);        % X^{intra}
        for k = 1 : size(X_intra_wo_j, 2)
            error_intra = TrainSet.X(:,j) - X_intra_wo_j(:,k);  
            Ew = Ew + error_intra * error_intra';               % Eq.(14)    
        end

        % calcuate Eb
        inter_class_index = find(TrainSet.y ~= j_class);
        X_inter = TrainSet.X(:, inter_class_index);             % X^{inter}

        for k = 1 : size(X_inter, 2)
            error_inter = TrainSet.X(:,j) - X_inter(:,k); 
        end
        Eb = Eb + error_inter * error_inter';                   % Eq.(13)
    end

    % calc
    Ew = Ew/train_num;                                          % Eq.(14)
    Eb = Eb/(train_num * (class_num-1));                        % Eq.(13)
    epsilon = 0.001;
    Ew = Ew + epsilon*eye(size(Ew,2));

    [V, ~] = eig(Eb, Ew);                                       % Eq.(18)
    for i = 1:reduce_dimension
        U(:,i) = V(:,size(V,2)+1-i);
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

        X_i = TrainSet.X_red(:, class_index);
        alpha_i = pinv(X_i' * X_i) *  X_i';
        H{i} = X_i * alpha_i;

        if verbose
            fprintf('# LDRC: calc Hi for class : %03d/%03d (samples: %03d)\n', i, class_num, length(class_index));
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
            dis(1, i) = norm(y - y_pred);
        end
        [~, label] = min(dis);
        identity(j) = label;        

        if verbose
            correct = (label == TestSet.y(1, j));
            fprintf('# LDRC: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', j, label, TestSet.y(1, j), correct);
        end          
    end
    
    
    % calculate accuracy
    correct_num = sum(identity == TestSet.y);    
    accuracy = correct_num / test_num;
end

