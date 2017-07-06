function  A_projected = spd_project(A)
% Projection algorithm onto symmetric positive definite (SPD) matrix
%
% Inputs:
%       A                   input matrix
% Output:
%       A_projected         projected matrix onto SPD
%
% Created by K.Yoshikawa and H.Kasai on June 23, 2017


    dim = size(A, 1);
    dd  = zeros(1, dim);
    v_t = zeros(1, dim*dim);
    iter_max = 100;
    

    % user the library from https://people.sc.fsu.edu/~jburkardt/m_src/jacobi_eigenvalue/jacobi_eigenvalue.html
    [v, d, ~, ~] = jacobi_eigenvalue (dim, A, iter_max);
    v = v(:)';
    d = d';

    
    for i = 0 : dim-1
        if d(1, i+1)>1e-4
            dd(1, i*dim+i+1) = d(1,i+1);
        else
            dd(1, i*dim+i+1) = 1e-4;
        end
        
        for j = 0 : dim-1
            v_t(1, j*dim+i+1) = v(1, i*dim+j+1);
        end
    end
    
    temp = mxm_jik(dim, dim, dim, v, dd);
    result = mxm_jik(dim, dim, dim, temp, v_t);
    
    A_projected = zeros(dim, dim);
    for i = 0 : dim-1
        for j = 0 : dim-1
            A_projected(i+1, j+1) = result(1, i*dim+j+1);
        end
    end
end


function [a] = mxm_jik(n1, n2, n3, b, c)
    a = zeros(1, n1*n1);

    for j = 0 : n3 -1
        for i = 0 : n1 -1
            a(1, i+j*n1+1) = 0.0;
        end
    end

    for j = 0 : n3 -1
        for i = 0 : n1 -1
            for k = 0 : n2 -1
                a(1, i+j*n1+1) = a(1, i+j*n1+1) + b(1, i+k*n1+1) * c(1, k+j*n2+1);
            end
        end
    end
end

