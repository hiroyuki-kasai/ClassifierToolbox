function [ accuracy ] = rcm_knn_classifier(TrainSet, TestSet, cov_type, cov_sub_type, distance_type, k)
        
    % set paramters
    train_num = size(TrainSet.X, 2);
    test_num = size(TestSet.X, 2);
    
    
    % set covariance matrices
    if strcmp(cov_type, 'GRCM')
        switch cov_sub_type
            case '1'
                Cg = TrainSet.GRCM1;
                Cp = TestSet.GRCM1;
            case '2'
                Cg = TrainSet.GRCM2;
                Cp = TestSet.GRCM2; 
            case '3'
                Cg = TrainSet.GRCM3;
                Cp = TestSet.GRCM3; 
            otherwise
                fprintf('Invalid GRCM cov_sub_type %s\n', cov_sub_type);
                return;
        end
    elseif strcmp(cov_type, 'RCM')
        switch cov_sub_type
            case '1'
                Cg = TrainSet.RCM1;
                Cp = TestSet.RCM1;
            case '2'
                Cg = TrainSet.RCM2;
                Cp = TestSet.RCM2; 
            case '3'
                Cg = TrainSet.RCM3;
                Cp = TestSet.RCM3; 
            case '4'
                Cg = TrainSet.RCM4;
                Cp = TestSet.RCM4; 
            case '5'
                Cg = TrainSet.RCM5;
                Cp = TestSet.RCM5;                 
            otherwise
                fprintf('Invalid RCM cov_sub_type %s\n', cov_sub_type);
                return;  
        end
    end
    
    
    % calculate distance
    distance.rho = zeros(test_num, train_num);
    distance.gallery_class = zeros(1, train_num);
    tmp_rho = zeros(1,5);

    fprintf('# RCM_kNN_Classifier calculates distance (%s:%s) ... ', cov_type, distance_type);
    for i = 1 : test_num
        for j = 1 : train_num 
            sum_rho = 0;
            for region = 1 : 5 
                tmp_rho(1,region) = calculate_spd_distance(Cg{region,j}, Cp{region,i}, distance_type);
                sum_rho = sum_rho + tmp_rho(1, region);
            end
            distance.rho(i,j) = sum_rho - max(tmp_rho);
        end
    end
    for i = 1 : train_num
        distance.gallery_class(1,i) = TrainSet.y(1,i);
    end
    fprintf('finished.\n');

    
    % divide sorted_distance from the top of 'k' and extract frequent class
    [sorted_distance, index] = sort(distance.rho,2);
    neighbors.index = zeros(test_num,k);
    neighbors.class = zeros(test_num,k);
    neighbors.most_frequent_class = zeros(test_num,1);

    for i = 1 : test_num
        neighbors.index(i,:) = index(i,1:k);
        neighbors.class(i,:) = distance.gallery_class(neighbors.index(i,:));
        neighbors.most_frequent_class(i,1) = mode(neighbors.class(i,:));
    end
    
    
    % calculate accurary
    %subtract_classes = neighbors.most_frequent_class(:,1) - Test.class';
    subtract_classes = neighbors.most_frequent_class(:,1) - TestSet.y';
    count = numel(subtract_classes(subtract_classes == 0));
    accuracy = count/test_num;
    
    fprintf('# RCM_kNN_Classifier: Accurary (%s:%s) = %5.2f\n', cov_type, distance_type, accuracy);
end

