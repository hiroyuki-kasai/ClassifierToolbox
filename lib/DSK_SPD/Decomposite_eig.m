function [decomp] = Decomposite_eig(X)

% input:  X is a cell structure, each containing a PSD matrix;
%         Y is a cell structure, each containing a PSD matrix;
%         index decides the set of eigenvalues to be included;
%

n_X = length(X);
if(n_X == 0)
    disp('X is empty, error!');
    return;
end

% extract eigenvalues and check PSD for input;

V_x = cell(n_X,1);
D_x = cell(n_X,1);

for i = 1:n_X
    [V,D] = eig(X{i});
    [Sorted_D,I] = sort(diag(D),'descend');
    
    V_x{i} = V(:,I);
    D_x{i} = abs(diag(Sorted_D));
end

decomp.V = V_x;
decomp.D = D_x;

end

