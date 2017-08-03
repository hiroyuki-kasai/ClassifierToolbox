function H_normalized = spd_normalization(H)

    n = 1 ./ sqrt(diag(H));
    H_normalized = diag(n) * H * diag(n);

end

