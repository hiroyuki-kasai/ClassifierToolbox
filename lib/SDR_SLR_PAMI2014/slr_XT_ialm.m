function [X T iter] = slr_XT_ialm(D, A, B, tau, delta, epsilon, maxIter)

%-------------------------------------------------------------------------------------
% Dec 2014
%
% by Jian Lai, jlai1@ntu.edu.sg
% solve the subproblem (23) in our PAMI paper
% min \tau ||X||_F^2 + \delta ||T||_1 s.t. D - A = BX + T.

% input
% D: training data, each column is a atom, with the size m*n
% A: current class-specific dictionary
% B: current nonclass-specific dictoinary
% tau, delta: the balancing parameters in (23)
% epsilon: the epsilonerance of the convergent checking
% maxIter: max iteration

% output
% X: the updated coefficient matrix of B
% T: the updated error matrix
%-------------------------------------------------------------------------------------

Dbar = D - A;

[m n] = size(Dbar);

X = zeros(n);
T = zeros(m, n);
Y = T;
BTB=B'*B;

mu = 1e-3;
mu_bar = 1e+6;
rho = 1.5;
iter = 0;
converged = false;

while ~converged

    iter = iter + 1;

    %optimize T
    Resi_e = Dbar - B*X + Y/mu;
    T = max(Resi_e - delta/mu, 0)+min(Resi_e + delta/mu, 0);             

    %optimize X
    Resi_x=B'*(Y+mu*(Dbar-T));
    X=(mu*BTB+2*tau*eye(n))\Resi_x;   

    %update Y1 Y2 and \mu
    Z = Dbar - B*X - T ;
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);

    %check convergence
    Zn=norm(Z,'fro');
    if Zn < epsilon
        converged = true;
    end    

    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end
