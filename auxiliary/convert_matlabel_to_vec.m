function label_vec  = convert_matlabel_to_vec(label_mat, num)
    label_vec = zeros(1, num);
    for i=1:num
        nonzero_idx = find(label_mat(:,i)>0);
        if length(nonzero_idx) > 1
            nonzero_idx = min(nonzero_idx);
        end
        label_vec(1, i) = nonzero_idx;
    end
end

