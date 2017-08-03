function [DR_TrainSet, DR_TestSet] = spd_dr(TrainSet, TestSet, newDim, metric, options)

    % Author:
    % - Mehrtash Harandi (mehrtash.harandi at gmail dot com)
    %
    % This file is provided without any warranty of
    % fitness for any purpose. You can redistribute
    % this file and/or modify it under the terms of
    % the GNU General Public License (GPL) as published
    % by the Free Software Foundation, either version 3
    % of the License or (at your option) any later version.
    
% Inputs:
%       TrainSet            train sets of size dxn, where d is dimension and n is number of sets 
%       TestSet             test sets of size dxn, where d is dimension and n is number of sets
%       newDim              new dimension of reduction
%       metric              metric (1: AIRM, 2: Stein)
%       options             options
% Output:
%       DR_TrainSet         dimensionality-reduced TrainSet
%       DR_TestSet          dimensionality-reduced TestSet
%
% References:
%       M. Harandi, M. Salzmann and R. Hartley, 
%       "From manifold to manifold: geometry-aware dimensionality reduction for SPD matrices,"
%       European Conference on Computer Vision (ECCV), 2014.
%
%
% Modified by H.Kasai on Aug. 03, 2017    

    if ~isfield(options, 'verbose')
        verbose = true;
    else
        verbose = options.verbose;
    end
    
    if ~isfield(options, 'graph_kw')
        graph_kw = 15;
    else
        graph_kw = options.graph_kw;
    end
    
    if ~isfield(options, 'graph_kb')
        graph_kb = 10;
    else
        graph_kb = options.graph_kb;
    end
    
    if ~isfield(options, 'maxiter')
        maxiter = 100;
    else
        maxiter = options.maxiter;
    end    
    

    train_num = length(TrainSet.y);
    test_num = length(TestSet.y);
    n = size(TrainSet.X, 1);
    r = newDim;

    %Generating graph
    G = generate_Graphs(TrainSet.X, TrainSet.y, graph_kw, graph_kb, metric);

    %- different ways of initializing, the first 10 features are genuine so
    %- the first initialization is the lucky guess, the second one is a random
    %- attempt and the last one is the worst possible initialization.

    % W = orth(rand(n, r));
    W = eye(n, r);
    % W = [zeros(n-r, r);eye(r)];

    % Create the problem structure.
    manifold = grassmannfactory(n, r);
    problem.M = manifold;

    % conjugate gradient on Grassmann
    covD_Struct.X = TrainSet.X;
    covD_Struct.y = TrainSet.y;
    covD_Struct.r = r;
    covD_Struct.n = n;
    covD_Struct.Metric_Flag = metric;
    covD_Struct.G = G;
    problem.costgrad = @(W) supervised_WB_CostGrad(W, covD_Struct);
    
    manopt_options.verbosity = verbose;
    manopt_options.maxiter = maxiter;
    W  = conjugategradient(problem, W, manopt_options);


    DR_TrainSet.X = zeros(newDim, newDim, train_num/2);
    for i = 1:train_num
        DR_TrainSet.X(:,:,i) = W' * TrainSet.X(:,:,i) * W;
    end
    DR_TrainSet.y = TrainSet.y;
    
    DR_TestSet.X = zeros(newDim, newDim, test_num);
    for i = 1:test_num
        DR_TestSet.X(:,:,i) = W' * TestSet.X(:,:,i) * W;
    end
    DR_TestSet.y = TestSet.y;
end




