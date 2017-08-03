function H_tensor_normalized = spd_tensor_normalization(H_tensor)

    dim = size(H_tensor, 1);
    len = size(H_tensor, 3);
    H_tensor_vec = zeros(dim*dim, len);
    H_tensor_normalized = zeros(dim, dim, len);
    
    for i = 1 : len
        H_tensor_vec(:, i) = vec(H_tensor(:, :, i));
        %H_tensor_normalized(:, :, i) = spd_normalization(H_tensor(:, :, i));
    end
    
    [H_tensor_vec, ~] = data_normalization(H_tensor_vec, [], 'std'); 
    
    for i = 1 : len
        H_tensor_normalized(:, :, i) = reshape(H_tensor_vec(:, i), [dim dim]);
    end    
    
end

