function accuracy = rsr_classifier(TrainSet, TestSet, options)
% Riemannian sparse representation-based (RSR) classifier
% This code is originally created by Mehrtash Harandi (mehrtash.harandi at gmail dot com)
% This code is modified by H. Kasai.
    
    
    % retrieve dimension of the SPD matrices
    dim = size(TrainSet.X_cov, 1); % dimension of the SPD matrices
    
    % calculate eigen decomposition
    train_decomp = Decomposite_eig_new(TrainSet);
    test_decomp = Decomposite_eig_new(TestSet);
    
    % learn alpha of kernel
    if ~options.original_alpha
        initial_alpha = 1*ones(1,dim); % the initial alpha corresponding to the original Stein kernel
        LB = 0.01*initial_alpha; 

        fmincon_opt = optimset('Algorithm','interior-point'); % run interior-point algorithm
        fmincon_opt.Display = 'iter';
        fmincon_opt.Display = 'off';
        fmincon_opt.MaxIter = 100;
        fmincon_opt.TolFun = 1e-5;
        %tic
        optimal_alpha = fmincon(@(alpha) objfun_ff_new(alpha,TrainSet.y,train_decomp, options.lambda,initial_alpha,options.obj_method,options.theta),initial_alpha,[],[],[],[],LB,[],[],fmincon_opt);
        %toc
    else
        optimal_alpha = ones(1,dim);
    end
       
    % compute the Stein divergence with the obtained adjustment parameter optimal_alpha
    S_test          = EigComp2SD_power_new(train_decomp, test_decomp,optimal_alpha); 
    S_train         = EigComp2SD_power_new(train_decomp, train_decomp,optimal_alpha);
    probeKernel     = exp(-1*options.theta*S_test); % compute the kernel
    galleryKernel   = exp(-1*options.theta*S_train);    

    % normalize dictionary
    KD = galleryKernel./repmat(sqrt(sum(galleryKernel.^2)),[size(galleryKernel,1) 1]);
    KX = probeKernel./repmat(sqrt(sum(probeKernel.^2)),[size(probeKernel,1) 1]);
    L1 = size(probeKernel, 2);

    [KD_U,KD_D,~] = svd(KD);    
    A = diag(sqrt(diag(KD_D)))*KD_U';
    D_Inv = KD_U*diag(1./sqrt(diag(KD_D)));
    KX = D_Inv'*KX;
    
    % perform lasso
    param.lambda = options.lambda;
    param.lambda2 =  0; 
    param.mode = 2;
    scX = full(mexLasso(KX,A,param));
 
    % classify 
    if strcmp(options.mode, 'src')
        qLabel = Classify_SRC(A, TrainSet.y, scX, KX, TestSet.y, options.verbose);
    elseif strcmp(options.mode, 'ip_linear')
        qLabel = Classify_Similarity(TrainSet.y, scX, KX, 'linear', TestSet.y, options.verbose);
    elseif strcmp(options.mode, 'ip_max')
        qLabel = Classify_Similarity(TrainSet.y, scX, KX, 'max', TestSet.y, options.verbose);
    end        

    % calculate accuracy
    accuracy = sum((qLabel-TestSet.y == 0))/L1;
end


