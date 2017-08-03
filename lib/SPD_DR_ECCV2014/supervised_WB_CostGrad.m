% Author:
% - Mehrtash Harandi (mehrtash.harandi at gmail dot com)
%
% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.

function [outCost,outGrad,covD_Struct] = supervised_WB_CostGrad(U,covD_Struct)
outCost = 0;
dF = zeros(size(U));

nPoints = length(covD_Struct.y);

I_r = eye(covD_Struct.r);
UXU = zeros(covD_Struct.r,covD_Struct.r,nPoints);
inv_UXU = zeros(covD_Struct.r,covD_Struct.r,nPoints);
for tmpC1 = 1:nPoints
    UXU(:,:,tmpC1) = U'*covD_Struct.X(:,:,tmpC1)*U;
    inv_UXU(:,:,tmpC1) = I_r/UXU(:,:,tmpC1);
end





for i = 1:nPoints
    X_i = covD_Struct.X(:,:,i);
    for j = 1:nPoints
        if (covD_Struct.G(i,j) == 0)
            continue;
        end
        
        X_j = covD_Struct.X(:,:,j);
        switch (covD_Struct.Metric_Flag)

            case 1
                %AIRM
                outCost = outCost + covD_Struct.G(i,j)*Compute_AIRM_Metric(UXU(:,:,i) , UXU(:,:,j));
                log_XY_INV = logm(UXU(:,:,i)*inv_UXU(:,:,j));
                
                dF = dF + 4*covD_Struct.G(i,j)*((X_i*U)*inv_UXU(:,:,i)  ...
                    -(X_j*U)*inv_UXU(:,:,j) )*log_XY_INV;
             case 2
                %Stein  metric
                outCost = outCost + covD_Struct.G(i,j)*Compute_Stein_Metric(UXU(:,:,i) , UXU(:,:,j));
                
                X_ij = 0.5*(X_i + X_j);
                dF = dF + covD_Struct.G(i,j)*(2*(X_ij*U)/(U'*X_ij*U)  ...
                    - (X_i*U)*inv_UXU(:,:,i) - (X_j*U)*inv_UXU(:,:,j));
            otherwise
                error('The metric is not implemented.');
        end %end switch
        
    end
end





outGrad = (eye(size(U,1)) - U*U')*dF;


end

