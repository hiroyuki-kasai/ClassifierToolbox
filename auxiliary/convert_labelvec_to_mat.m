function label_mat  = convert_labelvec_to_mat(label_vec, num, class_num)
    label_mat = zeros(class_num, num);
    for i=1:num
        label_mat(label_vec(i),i) = 1;
    end
end

