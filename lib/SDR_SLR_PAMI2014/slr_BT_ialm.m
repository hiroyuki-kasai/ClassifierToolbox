function [B T iter] = slr_BT_ialm(D, A, X, lambda, delta, epsilon, maxIter)

%-------------------------------------------------------------------------------------
% Dec 2014
%
% by Jian Lai, jlai1@ntu.edu.sg
% solve the subproblem (20) in our PAMI paper
% min \lambda ||B||_* + \delta ||T||_1 s.t. D - A = BX + T.

% input
% D: training data, each column is a atom, with the size m*n
% A: current class-specific dictionary
% X: current coefficents of B
% lambda, delta: the balancing parameters in (20)
% epsilon: the tolerance of the convergent checking
% maxIter: max iteration

% output
% B: the updated nonclass-specific dictionary
% T: the updated error matrix
%-------------------------------------------------------------------------------------

Dbar = D - A;

[m n] = size(Dbar);

% initialization

T = zeros(m, n);
B = zeros(m, n);
J = zeros(m, n);
Y1 = zeros(m, n);
Y2 = zeros(m, n);
XXT = X*X';
invB = inv(XXT+eye(size(X,1)));

mu = 1e-3;
rho = 1.5;
mu_bar = 1e+6;
iter = 0;
converged = false;

while ~converged

    iter = iter + 1;      
        
	%optmize J
    Resi_j = B+Y2./mu;
    [U S V] = svd(Resi_j, 'econ');
    diagS = diag(S);
    svp = length(find(diagS > lambda/mu));
    if svp>=1
        diagS = diagS(1:svp)-lambda/mu;
    else
        svp = 1;
        diagS = 0;
    end
    J = U(:,1:svp)*diag(diagS)*V(:,1:svp)';  
    
    %optimize B
    Resi_b = (Dbar-T)*X'+Y1*X'/mu-Y2/mu+J;
    B=Resi_b * invB;         

    %optimize T
    Resi_e = Dbar - B*X + Y1/mu;
    T = max(Resi_e - delta/mu, 0) + min(Resi_e + delta/mu, 0);               

    %update Y1 Y2 and \mu
    Z1 = Dbar - B*X - T ;
    Y1 = Y1 + mu*Z1;
    Z2 = B - J ;
    Y2 = Y2 + mu*Z2;
    mu = min(mu*rho, mu_bar);

    %check convergence
    Z1n = norm(Z1,'fro');
    Z2n = norm(Z2,'fro');        
    if Z1n < epsilon && Z2n < epsilon 
        converged = true;
    end    

    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end

B = J;   
