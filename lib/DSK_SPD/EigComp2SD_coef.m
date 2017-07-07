%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [S] = EigComp2SD_power(decompX,decompY,alpha)
% this function compute the Stein divergence with optimized alpha
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input parameters:
% alpha: the optimized adjustment parameters (as coefficient)
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
function [S,ds] = EigComp2SD_coef(decompX,decompY,alpha)
V_x = decompX.V;
D_x = decompX.D;
if(~isempty(decompY))
    V_y = decompY.V;
    D_y = decompY.D;
else
    V_y = [];
    D_y = [];
end

n_X = length(V_x);
if(n_X == 0)
    disp('X is empty, error!');
    return;
else
    dim = size(V_x{1},1);
    n_Y = length(V_y);
end

% extract eigenvalues and check PSD for input;

eig_X = zeros(n_X,dim);
for i = 1:n_X
    eig_X(i,:) = diag(D_x{i});
end

eig_X = abs(eig_X);
eig_X = log(repmat(alpha,n_X,1))+ log(eig_X);  %check
eig_X = sum(eig_X(:,:),2);

if(n_Y)
    eig_Y = zeros(n_Y,dim);
    for i = 1:n_Y
        eig_Y(i,:) = diag(D_y{i});
    end
    
    eig_Y = abs(eig_Y);
    eig_Y = log(repmat(alpha,n_Y,1)) + log(eig_Y);
    eig_Y = sum(eig_Y(:,:),2);
end

if(n_Y)
    XY_tude = cell(n_X,n_Y);
    det_XY = zeros(n_X,n_Y);
    for i = 1:n_X
        X_tude = Matrix_tude(V_x{i},D_x{i},alpha);
        for j = 1:n_Y
            Y_tude = Matrix_tude(V_y{j},D_y{j},alpha);
            XY_tude{i,j} = (X_tude + Y_tude ) ./ 2;
            det_XY(i,j) = det( XY_tude{i,j} );
        end
    end
    det_XY = log(det_XY);
    
    S = det_XY(:,:) - 0.5 * (repmat(eig_X,1,n_Y) + repmat(eig_Y',n_X,1));
else
    XX_tude = cell(n_X,n_X);
    det_XX = zeros(n_X,n_X);
    for i = 1:n_X
        X_i = Matrix_tude(V_x{i},D_x{i},alpha);
        for j = i:n_X
            X_j = Matrix_tude(V_x{j},D_x{j},alpha);
            XX_tude{i,j} = ( X_i + X_j ) ./ 2;
            det_XX(i,j) = det( XX_tude{i,j} );
            det_XX(j,i) = det_XX(i,j);
        end
    end
    det_XX = log(det_XX);
    S = det_XX(:,:) - 0.5 * (repmat(eig_X,1,n_X) + repmat(eig_X',n_X,1));
end
S(abs(S)<1e-10) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% compute ds %%%%%%%%
if(nargout == 2)
    nalpha = length(alpha);
    ds = cell(nalpha,1);
    
    if(n_Y)
        Inv_XY_tude = cell(n_X,n_Y);
        Inv_X_tude = cell(n_X,1);
        Inv_Y_tude = cell(n_Y,1);
        for i = 1:n_X
            Inv_X_tude{i} = V_x{i}*diag(1./diag(D_x{i}))*V_x{i}';
            for j = 1:n_Y
                Inv_XY_tude{i,j} = inv(XY_tude{i,j});
            end
        end
        for j = 1:n_Y
            Inv_Y_tude{j} = V_y{j}*diag(1./diag(D_y{j}))*V_y{j}';
        end
    else
        Inv_XX_tude = cell(n_X,n_X);
        Inv_X_tude = cell(n_X,1);
        for i = 1:n_X
            Inv_X_tude{i} = V_x{i}*diag(1./diag(D_x{i}))*V_x{i}';
            for j = i:n_X
                Inv_XX_tude{i,j} = inv(XX_tude{i,j});
            end
        end
    end
   
    for ialpha = 1:nalpha
        if(n_Y)
            ds_i = zeros(n_X,n_Y);
            dx_alphai = cell(n_X,1);
            dy_alphai = cell(n_Y,1);
            for i = 1:n_X
                dD_xi = zeros(nalpha,1);
                eig_x = diag(D_x{i});
                dD_xi(ialpha) = eig_x(ialpha).*alpha(ialpha)*log(eig_x(ialpha));
                dx_alphai{i} = V_x{i}*diag(dD_xi)*V_x{i}';
                
            end
            for j = 1:n_Y
                dD_yj = zeros(nalpha,1);
                eig_y = diag(D_y{i});
                dD_yj(ialpha) = eig_y(ialpha).*alpha(ialpha)*log(eig_y(ialpha));
                dy_alphai{j} = V_y{j}*diag(dD_yj)*V_y{j}';
            end
            for i = 1:n_X
                for j = 1:n_Y
                    ds_i(i,j) = 0.5*trace(Inv_XY_tude{i,j}*(dx_alphai{i}+dy_alphai{j})) - 0.5*(trace(Inv_X_tude{i}*dx_alphai{i})+trace(Inv_Y_tude{j}*dy_alphai{j}));
                end
            end
        else
            ds_i = zeros(n_X,n_X);
            dx_alphai = cell(n_X,1);
            
            for i = 1:n_X
                dD_xi = zeros(nalpha,1);
                eig_x = diag(D_x{i});
                dD_xi(ialpha) = eig_x(ialpha).*alpha(ialpha)*log(eig_x(ialpha));
                dx_alphai{i} = V_x{i}*diag(dD_xi)*V_x{i}';
                
            end
            for i = 1:n_X
                for j = i:n_X
                    ds_i(i,j) = 0.5*trace(Inv_XX_tude{i,j}*(dx_alphai{i}+dx_alphai{j})) - 0.5*(trace(Inv_X_tude{i}*dx_alphai{i})+trace(Inv_X_tude{j}*dx_alphai{j}));
                    ds_i(j,i) = ds_i(i,j);
                end
            end
        end
        ds{ialpha} = ds_i;
    end
    
end

end
function [M] = Matrix_tude(U,eig,alpha)
M = U*diag(diag(eig)'.*alpha)*U';
end
