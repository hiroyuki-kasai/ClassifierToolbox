function [dic_cell, train_time] = KSVD_trainer(params, train_images, train_labels)
 
    num_classes           = params.num_classes;       % number of classes in database
    alg_type              = params.alg_type;          % algortihm: 'KSVD','KKSVD'
    iter                  = params.iter;              % number of dictionary learning iterations
    card                  = params.card;              % cardinality of sparse representations
    

    train_cell = cell(1, num_classes);
    % divide the training set to different classes
    for i = 1:num_classes
        train_cell{i} = train_images(:,train_labels(i,:) == 1);
    end

    % initialize dictionary
    dic_cell = init_dictionary(params, train_images, train_labels, train_cell);

    % dictionary training
    if strcmp(alg_type, 'KSVD')
        train_tic = tic;
        fprintf('## KSVD Learner starts ... \nclass ');
        for i = 1:num_classes
            fprintf('%03d ', i);    
            params_ksvd = [];
            params_ksvd.data = train_cell{i};
            params_ksvd.Tdata = card;
            params_ksvd.iternum = iter;
            params_ksvd.initdict = dic_cell{i};
            params_ksvd.memusage = 'high';
            [dic_cell{i}] = ksvd(params_ksvd,'');
            if (mod(i, 10) == 0) && (i ~= num_classes)
                fprintf('\nclass ');
            end
        end
        fprintf('\n## KSVD Learner finished.\n');
        train_time = toc(train_tic);
    end
    
end

function dic_cell = init_dictionary(params, train_images, train_labels, train_cell)

    % ========================================================================
    % Author: Alona Golts (zadneprovski@gmail.com)
    % Date: 05-04-2016
    % 
    % Initialize dictionary before performing dictionary learning
    %
    % INPUT:
    % params - struct containing all classification params
    % train_cell      - cell containing the train input divided by classes
    %
    % OUTPUT:
    % dic_cell        - cell containing a dictionary initialization for each
    % class
    % ========================================================================

    init_dic       = params.init_dic;
    num_runs       = params.num_runs;
    num_classes    = params.num_classes;
    dict_size      = params.dict_size;
    sig_dim        = size(train_cell{1}, 1);
    dic_cell       = cell(1, num_classes);


    switch init_dic
        case 'random' % entirely random elements
            for i = 1:num_classes
                dic_cell{i} = randn(sig_dim,dict_size);
            end
        case 'partial' % initial columns in dictionary are some of the train samples
            for i = 1:num_classes
                if (num_runs == 1)
                    rng(i);
                end
                if ((size(train_cell{i},2)) >= dict_size)
                    ind = randperm(size(train_cell{i},2));
                    ind = ind(1:dict_size);
                else
                    ind = randperm(dict_size);
                    ind = mod(ind,size(train_cell{i},2)) + 1;
                end
                dic_cell{i} = train_cell{i}(:,ind) ;
                dic_cell{i} = dic_cell{i}.*repmat(1./sqrt(sum(dic_cell{i}.*dic_cell{i})),[sig_dim,1]);
            end
        case 'partial_new' % from LC-KSVD project
            for i = 1:num_classes            
                col_ids = find(train_labels(i,:)==1);
                data_ids = find(colnorms_squared_new(train_images(:,col_ids)) > 1e-6);   % ensure no zero data elements are chosen
                %    perm = randperm(length(data_ids));
                perm = 1:length(data_ids); 

                dic_cell{i} = train_images(:,col_ids(data_ids(perm(1:dict_size))));             
            end
    end

end