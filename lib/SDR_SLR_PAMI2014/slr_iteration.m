function [A B X E] = slr_iteration(D, labels, lambda, tau, eta, delta, epsilon, maxIter, maxOut)

% Dec 2014
%
% by Jian Lai, jlai1@ntu.edu.sg
% solve the equation (17) the supervised low rand decomposition in our PAMI paper
% min ||A||_* + \lambda ||B||_* + \tau ||X||_F^2 + \eta ||E||_1 s.t. D = A + BX + E.

% input
% D: training data, each column is a atom, with the size m*n
% labels: the training labels of D, with the size n*1
% lambda, tau and eta: the balancing parameters in equation (17)
% delta: the balancing parameter of T in equation (20) and (23)
% epsilon: the tolerance of the convergent checking
% maxIter: max iteration within each subproblem
% maxOut: max iteration of outer loop (defaul value is 4)

% output
% A: the class-specific dictionary
% B: the nonclass-specific dictionary
% X: the coefficient matrix of B
% E: the error matrix




% input checking
if nargin < 6
    error('Too few arguments') ;
end

if nargin < 7
    epsilon = 1e-4;
end

if nargin < 8
    maxIter = 1000;
end

if nargin < 9
    maxIter = 4;
end

% initializate A_i as (19) in our PAMI paper
unilabel = unique(labels);
numlabel = size(unilabel,2);
for classindex = 1:numlabel   
    classid = unilabel(classindex);
    Di = D(:,classid==labels);
    [U S V] = svd(Di, 'econ');
	Ai{classindex} = U(:, 1) * diag(S(1,1)) * V(:, 1)';    
end

% initialize of A
A = [];
for classindex = 1:numlabel
    A = [A Ai{classindex}];
end

% initialize X to an identity matrix
X = eye(size(A,2));

% initialize B to a zero matrix
B = zeros(size(D));

for outiter = 1:maxOut
    
    fprintf('The %d iteration of SLR\n', outiter);
    
    % solve the subproblem (20) in our PAMI paper
    [B, T] = slr_BT_ialm(D, A, X, lambda, delta, epsilon, maxIter);
    
    % solve the subproblem (23) in our PAMI paper
    [X, T] = slr_XT_ialm(D, A, B, tau, delta, epsilon, maxIter);
        
    % solve the subproblem (25) in our PAMI paper
    [A, E] = slr_AE_ialm(D, B, X, eta, epsilon, maxIter); 
        
end
