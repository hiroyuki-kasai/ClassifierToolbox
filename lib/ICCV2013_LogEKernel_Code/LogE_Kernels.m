function [ kernel_matrix ] = LogE_Kernels(data1, data2, param)
% LogE_Kernels(data1, data2, param)  computes the kernel matrix between
% data1 andd data2. The size of data1 is d by d by n1, and data1(:, :, i) denotes 
% the logarithm of covariance matrix obtained from  sample i.
% Param contains the parameters used for computing the
% kernel matrix (c.f. SetParams.m).
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

switch param.kernel
        
        case('Log-E poly.')
            % The scripts that run fast
            X1 = reshape(data1.X, size(data1.X, 1) * size(data1.X, 2), size(data1.X, 3))';
            X1 = bsxfun(@rdivide, X1, sqrt(sum(X1.^2, 2)));
            X2 = reshape(data2.X, size(data2.X, 1) * size(data2.X, 2), size(data2.X, 3))';
            X2 = bsxfun(@rdivide, X2, sqrt(sum(X2.^2, 2)));
            
            if isequal(data1.X, data2.X)
                dist_matrix = 1 - squareform(pdist(X1, 'cosine'));
            else
                dist_matrix = 1 - pdist2(X1, X2, 'cosine');
            end
            kernel_matrix =  dist_matrix .^ param.n;
            
%             %The scripts that run slowly
%             for i = 1:size(data1.X, 3)
%                 X = data1.X(:,:,i);
%                 X = X / norm(X, 'fro');
%                 for j = 1:size(data2.X, 3)
%                     Y = data2.X(:,:,j);
%                     Y = Y / norm(Y, 'fro');
%                     simi_matrix(i,j) = trace(X*Y);
%                 end   
%             end
%             kernel_matrix =  simi_matrix .^ param.n;

        case('Log-E exp.')
            % The scripts that run fast
            X1 = reshape(data1.X, size(data1.X, 1) * size(data1.X, 2), size(data1.X, 3))';
            X1 = bsxfun(@rdivide, X1, sqrt(sum(X1.^2, 2)));
            X2 = reshape(data2.X, size(data2.X, 1) * size(data2.X, 2), size(data2.X, 3))';
            X2 = bsxfun(@rdivide, X2, sqrt(sum(X2.^2, 2)));
            
            if isequal(data1.X, data2.X)
                dist_matrix = 1 - squareform(pdist(X1, 'cosine'));
            else
                dist_matrix = 1 - pdist2(X1, X2, 'cosine');
            end
            kernel_matrix =  exp(dist_matrix .^ param.n);
            
%             %The scripts that run slowly
%             for i = 1:size(data1.X, 3)
%                 X = data1.X(:,:,i);
%                 X = X / norm(X, 'fro');
%                 for j = 1:size(data2.X, 3)
%                     Y = data2.X(:,:,j);
%                     Y = Y / norm(Y, 'fro');
%                     simi_matrix(i,j) = trace(X*Y);
%                 end   
%             end
%             kernel_matrix =  exp(simi_matrix .^ param.n);

        case('Log-E Gauss.')
           % The scripts that run fast
            X1 = reshape(data1.X, size(data1.X, 1) * size(data1.X, 2), size(data1.X, 3))';
            X2 = reshape(data2.X, size(data2.X, 1) * size(data2.X, 2), size(data2.X, 3))';
            
             if isequal(data1.X, data2.X)
                dist_matrix = squareform(pdist(X1, 'euclidean'));
            else
                dist_matrix = pdist2(X1, X2, 'euclidean');
            end
             kernel_matrix =  exp(-param.Beta.*dist_matrix.^2);
             
%             %The scripts that run slowly
%             for i = 1:size(data1.X, 3)
%                 X = data1.X(:,:,i);
%                 for j = 1:size(data2.X, 3)
%                     Y = data2.X(:,:,j);
%                     dist_matrix(i, j) = norm(X-Y, 'fro');
%                 end   
%             end
%             kernel_matrix =  exp(-param.Beta.*dist_matrix.^2);
            
end
    





    