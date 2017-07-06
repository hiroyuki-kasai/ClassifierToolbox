function d = calculate_spd_distance(C1, C2, algorithm)

    switch algorithm
        case 'Frobenius'
            
            d = norm(C1- C2, 'fro'); 
                      
        case 'EV'
            
            eigenvalues = eig(C1, C2);

            % round to nearest decimal widh N digits to avoid numerical unstabilty
            %N = 8;
            %eigenvalues = round(eigenvalues, N);

            d = sqrt( sum( (log(eigenvalues)).^2) );
            
        case 'EV2'

            eps = 1e-6; % for numerical stabilty
            [~,D] = eig(C1 + eps,C2 + eps);
            if(min(diag(D)) < -eps)
                disp('error occurred in generalized eigenvalue computation');
                return; 
            end
            d = sum(log(diag(abs(D))).^2);            
            
        case 'Stein' % Stein distance
            
            d = log(det(0.5*(C1+C2))) - 0.5*log(det(C1*C2));
            
        case 'AIRM' % Affine invariant Riemannian metric
            
            sqrtC1 = sqrtm(C1);

            %invsqrtA = inv(sqrtC1);
            %d = norm(logm(invsqrtC1 * C2 * invsqrtA));

            d = norm(logm( (sqrtC1 \ C2) / sqrtC1), 'fro');           
            
        case 'LERM' % Log-Euclideian Riemannian metric
            
            d = norm(logm(C1) - logm(C2), 'fro'); 
                           
        otherwise
            d = [];
    end

end

