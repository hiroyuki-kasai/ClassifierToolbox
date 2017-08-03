%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [S] = EigComp2SD_power(decompX,decompY,alpha)
% this function compute the Stein divergence with optimized alpha
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input parameters:
% alpha: the optimized adjustment parameters (as power)
% decompX: The eigen decomposition of X
% decompY: The eigen decomposition of Y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output parameters:
% S: the Stein divergence matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jianjia Zhang, jz163@uowmail.edu.au Dec, 2014, all rights reserved
% For implementation details, please refer to: 
% "Learning Discriminative Stein Kernel for SPD Matrices and Its Applications." 
% arXiv preprint arXiv:1407.1974 (2014).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [S] = EigComp2SD_power_new(decompX,decompY,alpha)
    V_x = decompX.V;
    D_x = decompX.D;
    if(~isempty(decompY))
        V_y = decompY.V;
        D_y = decompY.D;
    else
        V_y = [];
        D_y = [];
    end

    %n_X = length(V_x); % HK
    n_X = size(V_x, 3);
    if(n_X == 0)
        disp('X is empty, error!');
        return;
    else
        dim = size(V_x,1);
        %n_Y = length(V_y);
        
        if ~isempty(V_y)        % HK
            n_Y = size(V_y,3);  % HK
        else                    % HK
            n_Y = 0;            % HK
        end
    end

    % extract eigenvalues and check PSD for input;

    eig_X = zeros(n_X,dim);
    for i = 1:n_X
        eig_X(i,:) = diag(D_x(:,:,i));
    end

    eig_X = abs(eig_X);
    eig_X = repmat(alpha,n_X,1).*log(eig_X);  %check
    eig_X = sum(eig_X(:,:),2);

    if(n_Y)
        eig_Y = zeros(n_Y,dim);
        for i = 1:n_Y
            eig_Y(i,:) = diag(D_y(:,:,i));
        end

        eig_Y = abs(eig_Y);
        eig_Y = repmat(alpha,n_Y,1).*log(eig_Y);
        eig_Y = sum(eig_Y(:,:),2);
    end

    % compute stein divergence;
    if(n_Y)

        det_XY = zeros(n_X,n_Y);
        for i = 1:n_X
            X_tude = Matrix_tude(V_x(:,:,i),D_x(:,:,i),alpha);
            for j = 1:n_Y
                Y_tude = Matrix_tude(V_y(:,:,j),D_y(:,:,j),alpha);

                det_XY(i,j) = det((X_tude + Y_tude ) ./ 2  );
            end
        end
        det_XY = log(det_XY);

        S = det_XY(:,:) - 0.5 * (repmat(eig_X,1,n_Y) + repmat(eig_Y',n_X,1));
    else

        det_XX = zeros(n_X,n_X);
        for i = 1:n_X
            X_i = Matrix_tude(V_x(:,:,i),D_x(:,:,i),alpha);
            for j = i:n_X
                X_j = Matrix_tude(V_x(:,:,j),D_x(:,:,j),alpha);
                det_XX(i,j) = det(( X_i + X_j ) ./ 2  );
                det_XX(j,i) = det_XX(i,j);
            end
        end
        det_XX = log(det_XX);
        S = det_XX(:,:) - 0.5 * (repmat(eig_X,1,n_X) + repmat(eig_X',n_X,1));
    end
    S(abs(S)<1e-10) = 0;

end


function [M] = Matrix_tude(U,eig,alpha)
    M = U*diag(diag(eig)'.^alpha)*U';
end
