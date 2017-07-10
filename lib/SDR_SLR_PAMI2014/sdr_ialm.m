function [ a x e ] = sdr_ialm(A, B, y, beta, gamma, epsilon, maxIter)

%-------------------------------------------------------------------------------------
% Dec 2014
%
% by Jian Lai, jlai1@ntu.edu.sg
% solve the equation (10) Sparse and Dense Hybrid Representation in our PAMI paper
% min ||a||_1 + \gamma ||x||_2^2 + \beta ||e||_1 s.t. y = Aa + Bx + e.

% input
% A: class-specific dictionary, each column is a atom
% B: nonclass-specific dictionary, each column is a atom
% beta: the balacing parameter of e
% gamma: the balancing parameter of x
% epsilon: the tolerance of the convergent checking
% maxIter: max iteration limit

% output
% a: the coefficients of A
% x: the coefficinets of B
% e: the error
%-------------------------------------------------------------------------------------

% check input
if nargin < 5
    error('Too few arguments') ;
end

if nargin < 6
    epsilon = 1e-6;
end

if nargin < 7
    maxIter = 1000;
end

% initialization
a=zeros(size(A,2),1);
x=zeros(size(B,2),1);
e=zeros(size(y));
phi=zeros(size(y));
[m, n]=size(A);
mu=1;
mu_bar = 1e+6;
rho=1.5;
iter = 0;
converged = false;
BTB=B'*B;

while ~converged

    iter = iter + 1;
    
	% optimize e
    Resi_e = y - A*a - B*x + phi/mu;
    e = max(Resi_e - beta/mu, 0)+min(Resi_e + beta/mu, 0); 
        
    %optimize x
    Resi_x = B'*( phi + mu * (y-A*a-e) );      
    x=(2*gamma*eye(n)+mu*BTB)\Resi_x;

    %optmize a
    Resi_a=y - B * x - e + phi/mu;
    a=SolveHomotopy(A, Resi_a, 'lambda', 1/mu, 'tolerance', 1e-5, 'stoppingcriterion', 3);
    
    %update \phi and \mu
    Z = y - A*a - B*x - e ;
    phi = phi + mu*Z;
    mu = min(mu*rho, mu_bar);
    
    %check convergence
    if norm(Z) < epsilon
        converged = true;
    end

    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
    
end

end

