%
% Riemannian Dictionary Learning and Sparse Coding Implementation.
% optimize for all the dictionary atoms in one shot using a power manifold.
% written by Anoop Cherian, Australian National University, anoop.cherian@gmail.com
%
% inputs:
% 1. train: a cell array, each cell is an SPD data matrix.
% 2. num_atoms: number of atoms to learn in the dictionary.
% 3. active_size: estimated number of active atoms (could be left as zero, as this parameter is currently not used in the code)
% 4. dataname: a suffix to be used when saving learned models. 
% 5. lambda: the sparsity regularization
% 6. key: a number to initilialize the random number generators (to reproduce the results).

% Outputs: 
% 1. BB, the learned dictionary, a cell array with each cell containing an SPD atom.
% 2. alpha, a matrix, the i-th column of which is the sparse code for the i-th data matrix in the dictionary BB.
% 3. obj, the objective after each iteration.
%
function [BB, alpha, obj] = Fast_Riem_DL(train, num_atoms, active_size, dataname, lambda, key)
addpath(genpath('./manopt'));

% prepare the data for fast processing.
% change train to X^{-1/2}
d = size(train{1},1);
inv_train = cellfun(@(x) inv(x), train, 'uniformoutput', false);
iTT = cat(3, inv_train{:});

BB = riem_dict_init(train, num_atoms);
plot_obj = 0;
obj = []; alpha = 0;
for t=1:10 % number of alternating minimizations.
    alpha = sparse_code(train, [], BB, lambda, 'riem', dataname, active_size, 'train', key);             	
    obj(t) = dl_objective(iTT, BB, alpha, 0);	
    BB = dict_learn(iTT, BB, alpha);	
    fprintf('Riem-DL iter=%d -- objective = %f\n', t, obj(t));
    if obj<1e-4 %convergence criteria.
        break;
    end
end

end

% use KMeans to initialize the dictionary, could also use LE-KMeans.
function BB = riem_dict_init(train, num_atoms)	
 	d = size(train{1},1); small = eye(d)*1e-10;    
    if size(train,1)~=1, train=train';end
    
 	train = cell2mat(cellfun(@(x) vec((x+small)), train, 'uniformoutput', false));
 	[assign, centers] = kmeans(train', num_atoms, 'emptyaction', 'drop', 'distance', 'sqeuclidean');
 	BB = arrayfun(@(i) (reshape(centers(i,:),[d,d])), 1:num_atoms, 'uniformoutput', false);
    
    BB = cellfun(@(x) x/trace(x), BB, 'uniformoutput', false);    
end

% dl part of the objective.
function dlobj = dl_objective(iTT, BB, alpha, should_display)
    c = 0; cc=[]; d=size(BB{1},1); delta=0; Id=eye(d);
    
	BBv = dict_vec(BB);
    for i = 1:size(iTT,3)
        P = iTT(:,:,i)*reshape(BBv*alpha(:,i),[d,d]);
        [U,E] = schur(P); E=diag(E); E(E<=1e-3) = 1;
        c = c + sum(log(E).^2);
        cc(i) = c;
    end
    if exist('should_display','var') && should_display==1
        figure(1); hold on; plot(cc,'color',rand(3,1)); hold off; drawnow;
    end
    dlobj = c/2 + delta*sum(sum((BBv-vec(Id)*ones(1,length(BB))).^2));
end

% total objective
function obj = total_objective(TT, BB, alpha, lambda)	
    obj = dl_objective(TT, BB, alpha);
    obj = obj + lambda*sum(alpha(:));
end

% dictionary learning using a power manifold of SPD manifolds.
% TT is the cat(3, sqrt_train{:});
function B = dict_learn(iTT, B_init, alpha)
    num_atoms = length(B_init);
    delta = 0.1; 
	n = size(iTT,3); d = size(iTT,1); Id = eye(d);

	M = sympositivedefinitefactory(d);
    %M = spectrahedronfactory(d, d); % spd with unit trace and full rank
    Mn = powermanifold(M, num_atoms);    
	problem.M = Mn;
	problem.cost = @objective;
	problem.egrad = @gradient;    

	% will use the training set and the cost.
	function c = objective(BB)        
        c = dl_objective(iTT, BB, alpha);
    end

    function g=gradient(BB)
        g = repmat({M.zerovec(BB{1})}, [num_atoms, 1]); % initialize gradient to zero cells.
        BBv = dict_vec(BB);
        P = zeros(d,d,n);
        for i=1:n
           S = reshape(BBv*alpha(:,i), [d,d]);         
           [U,E] = schur(iTT(:,:,i)*S); E=diag(E); E(E<1e-3)=1; 
           P(:,:,i) = U*diag(log(E))*U'/S;
        end
        P = reshape(P,[d^2, n]);
        for i=1:num_atoms
            g{i} = reshape(P*alpha(i,:)', [d,d]) + delta*(BB{i}-Id)*BB{i};
        end
    end
	
	options.maxiter = 50; % max num iterations of conjugate gradient.
    options.verbosity = 0;
    TT = cell(size(iTT,3),1);
    for t=1:size(iTT,3)
        TT{t} = inv(iTT(:,:,t));
    end
    B = conjugategradient(problem, B_init, options);
end
