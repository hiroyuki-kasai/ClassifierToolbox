function [accuracy, classify_results, classify_t] = KSVD_classifier(params, classifier_alg, test_images, test_labels, train_images, train_labels, dic_cell)

    if strcmp(classifier_alg, 'LKDL_type')
        
        fprintf('\n## %s classifier starts ... \nclass ', classifier_alg);

        classify_tic = tic; 
        
        %test = params.test_images;
        num_classes = params.num_classes;
        X = cell(num_classes,1);
        res = zeros(num_classes,size(test_images,2));
        classify_results = zeros(num_classes,size(test_images,2));
        
        for i = 1:num_classes
            fprintf('%03d ', i);   
            X{i} = omp(dic_cell{i}'*test_images, dic_cell{i}'*dic_cell{i}, params.card);
            res(i,:) = sqrt(sum((test_images - dic_cell{i}*X{i}).^2));
            if (mod(i, 10) == 0) && (i ~= num_classes)
                fprintf('\nclass ');
            end
        end

        [~,min_ind] = min(res,[],1);
        lin_ind = sub2ind(size(res),min_ind,1:size(res,2));
        classify_results(lin_ind) = 1;
        diff = sum(abs(classify_results - test_labels));
        accuracy = sum(diff==0)/length(diff);
        % disp([params.alg_type,': ',num2str(accuracy)]);
        classify_t = toc(classify_tic);    %% classifcation time
        fprintf('\n## %s classifier finished ( %d [sec]).\n', classifier_alg, classify_t);
        
    elseif  strcmp(classifier_alg, 'LC_KSVD_type')
        
        fprintf('\n## %s classifier starts ... \n', classifier_alg);

        classify_tic = tic;        
        num_classes     = params.num_classes;       % number of classes in database   
        dict_size       = params.dict_size;
        iter            = params.iter;              % number of dictionary learning iterations        
        cardinality     = params.card;              % cardinality of sparse representations
        dim             = size(test_images, 1);
        test_num        = size(test_images,2);
        
        % concatinate dictionaries
        dic_all = zeros(dim, dict_size * num_classes);
        for i=1:num_classes
            start_pos   = dict_size * (i-1) + 1;
            end_pos     = dict_size * i;
            dic_all(:,start_pos:end_pos) = dic_cell{i};
        end        
        
        % set paramters for display
        if test_num < 100
            dis_cycle = 1;
        elseif test_num < 1000
            dis_cycle = 10;
        else
            dis_cycle = 100;
        end

        % set paramters for KSVD
        params_ksvd             = [];
        params_ksvd.data        = train_images;
        params_ksvd.Tdata       = cardinality; % spasity term
        params_ksvd.iternum     = iter;
        params_ksvd.memusage    = 'high';
        params_ksvd.dict_size   = dict_size * num_classes;
        % normalization
        params_ksvd.initdict    = normcols(dic_all);

        % ksvd process
        fprintf('# calculate sparse code by K-SVD using the whole dictionary \n');
        [~, Xtemp, ~] = ksvd(params_ksvd,'');

        % learning linear classifier parameters
        %W = inv(Xtemp*Xtemp'+eye(size(Xtemp*Xtemp'))) * Xtemp*train_labels';
        W = (Xtemp*Xtemp'+eye(size(Xtemp*Xtemp'))) \ (Xtemp*train_labels');
        W = W';

        % sparse coding
        G = dic_all'*dic_all;
        Gamma = omp(dic_all' * test_images, G, cardinality);

        % classify process
        errnum = 0;
        err = [];
        prediction = [];
        classify_results = zeros(num_classes, test_num);
        dis_count = 0;
        
        fprintf('# %s classifier starts for each test sample... \nsample ', classifier_alg);
        for i=1:test_num
            if mod(i, dis_cycle) == 0
                fprintf('%04d ', i); 
                dis_count = dis_count + 1;
                
                if (dis_count > 0) && (mod(dis_count, 10) == 0) && (i ~= test_num)
                    fprintf('\nsample ');
                end                 
            end 
            
            spcode = Gamma(:,i);
            score_est =  W * spcode;
            score_gt = test_labels(:,i);
            [maxv_est, maxind_est] = max(score_est);  % classifying
            [maxv_gt, maxind_gt] = max(score_gt);
            classify_results(maxind_est, i) = 1;
            prediction = [prediction maxind_est];
            if(maxind_est~=maxind_gt)
                errnum = errnum + 1;
                err = [err;errnum i maxind_gt maxind_est];
            end
        end
        
        accuracy = (size(test_images,2)-errnum)/size(test_images,2);  
        classify_t = toc(classify_tic);
        
        fprintf('\n## %s classifier finished ( %d [sec]).\n', classifier_alg, classify_t);
    end