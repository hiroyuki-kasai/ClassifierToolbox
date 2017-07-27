function [TrainSet, TestSet, train_num, test_num, class_num] = reduce_dataset(TrainSet, TestSet, max_class_num, max_train_samples, max_test_samples)

    TrainSetSmall.X = [];
    TrainSetSmall.y = [];
    TestSetSmall.X = [];
    TestSetSmall.y = [];
    
    original_class_num = length(unique(TrainSet.y));
    if max_class_num > original_class_num
        max_class_num = original_class_num;
    end
    
    for i=1:max_class_num
        class_index = find(TrainSet.y == i);
        if length(class_index) > max_train_samples
            %class_index = class_index(1: max_samples);
            class_index_rnd = randperm(length(class_index));
            class_index_tmp = class_index(class_index_rnd);
            class_index = class_index_tmp(1: max_train_samples);
        end
        TrainSetSmall.X = [TrainSetSmall.X TrainSet.X(:, class_index)];
        TrainSetSmall.y = [TrainSetSmall.y TrainSet.y(1, class_index)];

        class_index = find(TestSet.y == i);
        if length(class_index) > max_test_samples
            %class_index = class_index(1: max_samples);
            class_index_rnd = randperm(length(class_index));
            class_index_tmp = class_index(class_index_rnd);
            class_index = class_index_tmp(1: max_test_samples);         
        end    
        TestSetSmall.X = [TestSetSmall.X TestSet.X(:, class_index)];
        TestSetSmall.y = [TestSetSmall.y TestSet.y(1, class_index)];    
    end
    clear TrainSet;
    clear TestSet;
    TrainSet = TrainSetSmall;
    TestSet = TestSetSmall;
    train_num = size(TrainSetSmall.X, 2);
    test_num = size(TestSetSmall.X, 2);
    class_num = length(unique(TrainSet.y));

end

