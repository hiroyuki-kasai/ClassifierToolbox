function code=sparse_code_wrapper(test)
    tic;
	n=length(test.X); dim=size(test.X{1},1); 
    if iscell(test.B)        
        code=zeros(length(test.B),n);
    else
        code = zeros(size(test.B,2),n);
    end
	if ~isfield(test,'gamma')
		test.gamma=0;
	end
	for i=1:n
		if isfield(test,'X')
			out = sdivLasso(test.B, test.X{i}, test.lambda, test.numiter, test.algo, test.gamma);
			code(:,i) = out.x;
		end	
	end
end
