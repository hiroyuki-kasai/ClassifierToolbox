function [A E iter] = slr_AE_ialm(D, B, X, eta, epsilon, maxIter)

%-------------------------------------------------------------------------------------
% Dec 2014
%
% by Jian Lai, jlai1@ntu.edu.sg
% solve the subproblem (25) in our PAMI paper
% min ||A||_* + \eta ||E||_1 s.t. D - BX = A + E.

% input
% D: training data, each column is a atom, with the size m*n
% B: current nonclass-specific dictoinary
% X: current coefficents of B
% eta: the balancing parameters in (25)
% epsilon: the epsilonerance of the convergent checking
% maxIter: max iteration

% output
% A: the updated class-specific dictionary A
% E: the updated error matrix
%-------------------------------------------------------------------------------------

Dbar = D - B*X;

[m n] = size(Dbar);

% initialize
Y = zeros(m, n);
A = zeros( m, n);
E = zeros( m, n);

mu = 1e-3;
mu_bar = 1e+6;
rho = 1.5;
iter = 0;
converged = false;

while ~converged       
    
    iter = iter + 1;
    
    % optimize E
    temp_T = Dbar - A + Y/mu;
    E = max(temp_T - eta/mu, 0) + min(temp_T + eta/mu, 0);

    % optimize A
    [U S V] = svd(Dbar - E + Y/mu, 'econ');
    diagS = diag(S);
    svp = length(find(diagS > 1/mu));
    A = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)';    
    
    % update Y and \mu
    Z = Dbar - A - E;
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
        
    %check convergence 
    stopCriterion = norm(Z, 'fro') / norm(Dbar, 'fro');
    if stopCriterion < epsilon
        converged = true;
    end     
    
    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end
