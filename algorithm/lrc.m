function accuracy = lrc(TrainSet, TestSet, test_num, class_num, options)
% Linear regression classificatoin (LRC) algorithm
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
%       I. Nassem, M. Bennamoun
%       'Linear regression for face recognition,'
%       IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.32, no.11, 2010.
%
%
% Created by H.Kasai on July 03, 2017


    % extract options
    if ~isfield(options, 'verbose')
        verbose = false;
    else
        verbose = options.verbose;
    end

    dim = size(TrainSet.X, 1);
    
    % prepare predicted label array
    identity = zeros(1, test_num);    

    if dim < 5000 

        % prepare projection matrix (hat matrix)
        H = cell(1, class_num);
        for i = 1 : class_num
            class_index = find(TrainSet.y == i);

            X_i = TrainSet.X(:, class_index);               % Eq.(1)
            beta_i = (X_i' * X_i) \ X_i';                   % Eq.(3) exept y (beta_i = inv(X_i' * X_i) * X_i')
            H{i} = X_i * beta_i;                            % Eq.(4) exept y

            if options.verbose
                fprintf('# LRC: class : %03d (samples: %03d)\n', i, length(class_index));
            end
        end


        % predict the class
        for j = 1 : test_num

            dis = zeros(1, class_num);
            y = TestSet.X(:, j);
            for i = 1 : class_num
                y_pred = H{i} * y;
                dis(1, i) = norm(y - y_pred);               % Eq.(5)
            end
            [~, label] = min(dis);                      % Eq.(6)
            
            identity(j) = label;
            
            if verbose
                correct = (label == TestSet.y(1, j));
                fprintf('# LRC: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', j, label, TestSet.y(1, j), correct);
            end             
        end
        
    else
        % This case avoids large-memory consumption for storing H_i

        % prepare distance_array
        distance_array = zeros(test_num, class_num);
        
        % calcualte distance for all test sets in each class
        for i = 1 : class_num
            
            % calculate projection matrix (hat matrix)
            class_index = find(TrainSet.y == i);
            X_i = TrainSet.X(:, class_index);               % Eq.(1)
            beta_i = (X_i' * X_i) \ X_i';                   % Eq.(3) exept y (beta_i = inv(X_i' * X_i) * X_i')
            H_i = X_i * beta_i;                             % Eq.(4) exept y
            
            for j = 1 : test_num
                y = TestSet.X(:, j);
                y_pred = H_i * y;
                distance_array(j, i) = norm(y - y_pred);    % Eq.(5)
            end            
            
            if options.verbose
                fprintf('# LRC: class : %03d/%03d (samples: %03d)\n', i, class_num, length(class_index));
            end
        end
        
        
        % predict the class
        for j = 1 : test_num
            dis = distance_array(j, :);
            [~, label] = min(dis);                            % Eq.(6)
            
            identity(j) = label;     
            
            if verbose
                correct = (label == TestSet.y(1, j));
                fprintf('# LRC: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', j, label, TestSet.y(1, j), correct);
            end              
        end        
    end
    
    
    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num / test_num;
end

