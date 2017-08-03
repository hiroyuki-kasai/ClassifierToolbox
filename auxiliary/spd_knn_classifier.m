function [accurarcy] = spd_knn_classifier(TrainSet, TestSet, metric)

    test_num = length(TestSet.y);

    if metric == 1
        %AIRM
        pair_dist = Compute_AIRM_Metric(TestSet.X, TrainSet.X);
    elseif metric == 2
        %Stein
        pair_dist = Compute_Stein_Metric(TestSet.X, TrainSet.X);
    else
        error('the metric is not defined');
    end

    [~, minIDX] = min(pair_dist);
    y_hat = TestSet.y(minIDX);
    accurarcy = sum(TestSet.y == y_hat)/test_num;    
end

