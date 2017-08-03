function accuracy = svm_classifier(TrainSet, TestSet, train_num, test_num, class_num, options)
% SVM classification (SRC) algorithm
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
    
    
    % generate eigenface
    if eigenface    
        [disc_set, ~, ~] = Eigenface_f(TrainSet.X, eigenface_dim);
        
        % project on subspace
        TrainSet.X  =  disc_set' * TrainSet.X;
        TestSet.X   =  disc_set' * TestSet.X;
    end

    % generate SVM models for each class
	models = cell(1, class_num);
	for i = 1 : class_num
        % create label data 
        label = zeros(train_num, 1) - 1;    % initialize as '-11' for all classes
        index = find(TrainSet.y == i);
        label(index) = 1;                   % set '1' for this class (i)
		
		% generate svm models
        models{i} = fitcsvm(TrainSet.X', label, 'KernelFunction','RBF', 'KernelScale','auto');
    end
    
    
    % prepare predicted label array
    identity = zeros(1, test_num);
    
    for i = 1 : test_num

        % prepare prediction result array
		prediction = zeros(1, class_num);
        
		for j = 1 : class_num
            % do SVM precition (classification)
            [prediction(j), ~] = predict(models{j}, TestSet.X(:,i)');
        end
        
        % calculate the predicted label
		label = find(prediction == 1);
        if isempty(label)
            label = -1;
        elseif length(label) > 1
            label = label(1);
        end
        identity(i) = label;
        
        if verbose
            correct = (label == TestSet.y(1, i));
            fprintf('# SVM: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', i, label, TestSet.y(1, i), correct);
        end           

    end

    % calculate accuracy
    correct_num = sum(identity == TestSet.y);
    accuracy = correct_num/test_num;
end