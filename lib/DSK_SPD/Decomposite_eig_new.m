function [decomp] = Decomposite_eig_new(M)

% input:  X is a cell structure, each containing a PSD matrix;
%         Y is a cell structure, each containing a PSD matrix;
%         index decides the set of eigenvalues to be included;
%
dim = size(M.X_cov,1);
n_X = size(M.X_cov,3);
if(n_X == 0)
    disp('X is empty, error!');
    return;
end

% extract eigenvalues and check PSD for input;

V_x = zeros(dim,dim,n_X);
D_x = zeros(dim,dim,n_X);

for i = 1:n_X
    [V,D] = eig(M.X_cov(:,:,i));
    Sorted_D = sort(diag(D),'descend');
    
    V_x(:,:,i) = fliplr(V(:,:));
    D_x(:,:,i) = abs(diag(Sorted_D));
end

decomp.V = V_x;
decomp.D = D_x;

end

