function [spcode,splcode] = sparse_code(data, labels, B, lambda, spmethod, dataname, active_size, tt, key, bs,gamma)
splcode = labels;
if nargin<10
   bs = 500;
end
	switch spmethod
		case 'riem'
			spcode = sparse_code_internal(data, B, lambda, 1, bs);
	end
end

% this code is written for multicore utility. However, multicore is currently not used in this package.
function spcode = sparse_code_internal(X, B, lambda, algo, bs, gamma)
	n=length(X); m=length(B); spcode = zeros(m,n); numiter=500; small=eye(size(X{1},1))*1e-7; 
	if nargin <5
		bs=100;
    end
	if nargin<6
		gamma=0;
	end

	nb=ceil(n/bs);
	for i=1:nb
		test{i}.B = B;
		for j=1:bs
			if ((i-1)*bs+j) > n
				break;
			end
			test{i}.X{j} = X{(i-1)*bs+j}+small;
		end
		test{i}.lambda = lambda;
		test{i}.numiter = numiter;
		test{i}.algo =algo;
		test{i}.idx = i;
		test{i}.gamma=gamma;
	end

	for i=1:length(test)
		spcode_cell{i} = sparse_code_wrapper(test{i});
        fprintf('# sparse code calculation for test data set: %d\n', i)
    end

	spcode = zeros(m,n); cnt = 0;		
	for i=1:nb
		xx = spcode_cell{i};
		if size(xx,2)==bs
			spcode(:,cnt+1:cnt+bs) = xx;%spcode_cell{i};
		else
			spcode(:,cnt+1:cnt+size(xx,2)) = xx;
		end
		cnt = cnt + size(xx,2);
	end
end

