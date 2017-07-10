function d = calculate_distance(a, b, algorithm)

    switch algorithm
        case 'Frobenius'
            
            d = norm(a - b, 'fro'); 
                      
        case 'Angle'
            
            cos = a' * b / (norm(a) * norm(b));
            
            d = -cos;
            
        otherwise
            d = [];
    end

end

