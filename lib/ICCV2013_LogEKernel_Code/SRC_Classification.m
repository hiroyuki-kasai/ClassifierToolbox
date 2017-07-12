function [test_labels min_val] =SRC_Classification(A, dict_labels, test_sparse_codes, test_samples)
% SRC_Classification(A, dict_labels, test_sparse_codes, test_samples) implements
% the sparse representation-based classification algorithm as described in
%
%J.Wright, A. Y. Yang, A., S. S. Sastry, Y. Ma. Robust Face Recognition via Sparse Representation. IEEE T. PAMI, 31(2): 210-227, 2009
%
% A denotes the dictionary the size of which is n by m, where n and m are the dimension and number of atoms, respectively.
% dict_labels corresponds to the categories labels of the atoms.
% test_sparse_codes denotes the sparse codes of the test samples, and its size is m by k, where k is the number of test samples.
% test_samples denotes the test samples, and its size is n by k.
%
% Please cite the following paper if you use the code:
%
% Peihua Li,  Qilong Wang, Wangmeng Zuo, and Lei Zhang. Log-Euclidean Kernels for Sparse 
% Representation and Dictionary Learning. IEEE Int. Conf. on Computer Vision (ICCV), 2013.
%
% For questions,  please conact:  Qilong Wang  (Email:  wangqilong.415@163.com), 
%                                               Peihua  Li (Email: peihuali at dlut dot edu dot cn) 
%
% The software is provided ''as is'' and without warranty of any kind,
% experess, implied or otherwise, including without limitation, any
% warranty of merchantability or fitness for a particular purpose.

num_test           = size(test_samples, 2);
num_categories = max(dict_labels);

category_idx = cell(num_categories, 1);
for i = 1:num_categories
    category_idx{i} = find(dict_labels==i);
end

residual_error = zeros(num_categories, num_test);
for i = 1:num_categories
    residual_y = test_samples - A(:, category_idx{i}) * test_sparse_codes(category_idx{i}, :);
    residual_error(i, :) = sum(residual_y .* residual_y, 1);
end
[min_val test_labels] = min(residual_error, [], 1);





