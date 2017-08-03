function accuracy = rsrc_classifier(TrainSet, TestSet, src_type, options)
% (Extended) Riemannian sparse coding classification (RSRC) algorithm
%
% Inputs:
%       TrainSet            train sets of size dxn, where d is dimension and n is number of sets 
%       TestSet             test sets of size dxn, where d is dimension and n is number of sets
%       src_type            Which extension version is activated. 
%       options             options
% Output:
%       accuracy            classification accurary
%
%
% Created by H. Kasai on July 11, 2017.
% Modified by H. Kasai on Aug. 3, 2017.
%
%
% Note that this code partially borrow the sparse coding code wirtten by A. Cherian.
% http://users.cecs.anu.edu.au/~cherian/code/rspdl.tar.gz
%
%   References:
%           A. Cherian, and S. Suvrit,
%           "Riemannian dictionary learning and sparse coding for positive definite matrices,"
%           IEEE Transactions on Neural Networks and Learning Systems, 2016.
% 
% Note that this code uses the geometry mean calculation code (KMeans_SPD_Matrices package) 
% wirtten by H. Salehian.
% https://jp.mathworks.com/matlabcentral/fileexchange/46343-kmeans-spd-matrices-zip

    % extract options
    if ~isfield(options, 'verbose')
        verbose = 1;
    else
        verbose = options.verbose;
    end
    
    if ~isfield(options, 'lambda')
        lambda = 1e-3;
    else
        lambda = options.lambda;
    end    

    if ~isfield(options, 'normalization')
        normalization = false;
    else
        normalization = options.normalization;
    end    
    
    if strcmp(src_type, 'RSRC')
        name = 'R-SRC';
    else
        error('Invalid src_type');
    end  
    
    train_num = length(TrainSet.y);
    test_num = length(TestSet.y);
    class_num = length(unique(TestSet.y));
    dim = size(TrainSet.X_cov, 1);

    % normalization
    if normalization
        TrainSet.X_cov = spd_tensor_normalization(TrainSet.X_cov);
        TestSet.X_cov = spd_tensor_normalization(TestSet.X_cov); 
    end    
    
    % format
    TrainSet.X_cov_cell = cell(1, train_num);    
    for i = 1 : train_num
        TrainSet.X_cov_cell{i} = TrainSet.X_cov(:, :, i);
        TrainSet.X_cov_vec(:, i) = vec(TrainSet.X_cov(:, :, i));
    end        
    TestSet.X_cov_cell = cell(1, test_num);
    for i = 1 : test_num
        TestSet.X_cov_cell{i} = TestSet.X_cov(:, :, i);
    end
    

    % calculate sparse code (alpha)
    alpha = sparse_code_internal(TestSet.X_cov_cell, TrainSet.X_cov_cell, lambda, 1);

    % prepare class array
    classes = unique(TrainSet.y); 
    
    % prepare predicted label array
    identity_red    = zeros(1, test_num);    
    identiti_sc_sum = zeros(1, test_num);  
    identiti_sc_max = zeros(1, test_num);    
    
    for i = 1 : test_num
        
        residuals   = zeros(1, class_num);
        sc_sum      = zeros(1, class_num);
        sc_max      = zeros(1, class_num);
        
        for j = 1 : class_num

            idx = find(TrainSet.y == classes(j));
            alpha_class_j = alpha(idx, i); 

            % calculate residual
            Q = TestSet.X_cov(:, :, i) \ reshape(TrainSet.X_cov_vec(:, idx) * alpha_class_j, [dim, dim]);
            [U, E] = schur(Q); 
            E = diag(E); 
            %E(E<=1e-3) = 1;
            
            residuals(j) = sum(log(E).^2);
            sc_sum(j) = sum(alpha_class_j);
            [sc_max(j), ~] = max(alpha_class_j);
        end
        
        % calculate label
        [~, label] = min(residuals); 
        identity_red(i) = classes(label);
        [~, label] = max(sc_sum); 
        identiti_sc_sum(i) = classes(label);
        [~, label] = max(sc_max); 
        identiti_sc_max(i) = classes(label);        
        
        if verbose > 1
            correct = (identity_red(i) == TestSet.y(1, i));
            fprintf('# %s:test:%03d, \tResidual: predict class: %03d --> ground truth :%03d (%d)\n', name, i, label, TestSet.y(1, i), correct);
            correct = (identiti_sc_sum(i) == TestSet.y(1, i));
            %fprintf('                  \tSC_sum:   predict class: %03d --> ground truth :%03d (%d)\n', label, TestSet.y(1, i), correct);
            correct = (identiti_sc_max(i) == TestSet.y(1, i));
            %fprintf('                  \tSC_max:   predict class: %03d --> ground truth :%03d (%d)\n', label, TestSet.y(1, i), correct);            
        end         
    end

    % calculate accuracy
    correct_num_src = sum(identity_red == TestSet.y);
    accuracy.residual = correct_num_src/test_num;
    correct_num_sc_sum = sum(identiti_sc_sum == TestSet.y);
    accuracy.sc_sum = correct_num_sc_sum/test_num;    
    correct_num_sc_max = sum(identiti_sc_max == TestSet.y);
    accuracy.sc_max = correct_num_sc_max/test_num; 
    
    if verbose > 0
        fprintf('# %s Accurarcy: src:%5.3f(=%d/%d), sc_sum:%5.3f(=%d/%d) sc_max:%5.3f(=%d/%d)\n', name, ...
                accuracy.residual, correct_num_src, test_num, ...
                accuracy.sc_sum, correct_num_sc_sum, test_num, ...
                accuracy.sc_max, correct_num_sc_max, test_num);        
    end
end


% The following code is written by Anoop Cherian.
% Implementation of Riemannian Dictionary Learning and Spares Coding described in this paper:
% Cherian, Anoop, and Suvrit Sra. "Riemannian Dictionary Learning and Sparse Coding for Positive Definite Matrices." 
% arXiv preprint arXi v:1507.02772 (2015).
% Please cite the above paper if you use this code. 
%
% Copyright (c) 2016, Anoop Cherian
%
% this code is written for multicore utility. However, multicore is currently not used in this package.
function spcode = sparse_code_internal(X, B, lambda, algo, bs, gamma)
	n=length(X); 
    m=length(B); 
    numiter=500; 
    small=eye(size(X{1},1))*1e-7; 
    
	if nargin <5
		bs=100;
    end
	if nargin<6
		gamma=0;
    end

    
	nb=ceil(n/bs);
    test = cell(1, nb);
	for i=1:nb
		test{i}.B = B;
		for j=1:bs
			if ((i-1)*bs+j) > n
				break;
			end
			test{i}.X{j} = X{(i-1)*bs+j}+small;
		end
		test{i}.lambda = lambda;
		test{i}.numiter = numiter;
		test{i}.algo =algo;
		test{i}.idx = i;
		test{i}.gamma=gamma;
	end

	for i=1:length(test)
		spcode_cell{i} = sparse_code_wrapper(test{i});
        fprintf('# sparse code calculation for test data set: %d/%d\n', i, length(test))
    end

	spcode = zeros(m,n); 
    cnt = 0;		
	for i=1:nb
		xx = spcode_cell{i};
		if size(xx,2)==bs
			spcode(:,cnt+1:cnt+bs) = xx;%spcode_cell{i};
		else
			spcode(:,cnt+1:cnt+size(xx,2)) = xx;
		end
		cnt = cnt + size(xx,2);
	end
end



