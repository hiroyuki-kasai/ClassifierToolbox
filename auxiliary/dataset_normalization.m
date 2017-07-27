function [train, train_l, test, test_l] = dataset_normalization(normalization_method, train, train_l, test, test_l)
    
    switch normalization_method
        case 'mean_std'
            train = train - repmat(mean(train),[size(train,1),1]);
            train = train./repmat(sqrt(sum(train.^2)),[size(train,1),1]);
            test = test - repmat(mean(test),[size(test,1),1]);
            test = test./repmat(sqrt(sum(test.^2)),[size(test,1),1]);
        case 'mean'
            train = train - repmat(mean(train),[size(train,1),1]);
            test = test - repmat(mean(test),[size(test,1),1]);
        case 'none'
            % Do nothing.
        case 'std'
            train = train./repmat(sqrt(sum(train.^2)),[size(train,1),1]);
            test = test./repmat(sqrt(sum(test.^2)),[size(test,1),1]);
        case 'sample_mean_per_class'
            classes = unique(train_l);
            class_num = length(classes);
            for j = 1 : class_num
                train_idx = find(train_l == classes(j));    
                class_mean = mean(train(:,train_idx), 2);
                train(:,train_idx) = train(:,train_idx) - repmat(class_mean, 1, length(train_idx));    
                
                test_idx = find(test_l == classes(j));    
                class_mean = mean(test(:,test_idx), 2);
                test(:,test_idx) = test(:,test_idx) - repmat(class_mean, 1, length(test_idx));                    
            end 
            
        otherwise
            
    end  
    
end
