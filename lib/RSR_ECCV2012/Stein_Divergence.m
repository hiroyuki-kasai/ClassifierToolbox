% Author:
% - Mehrtash Harandi (mehrtash.harandi at gmail dot com)
%
% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.

function outS = Stein_Divergence(X1,X2)
l1 = size(X1,3);
l2 = size(X2,3);

outS = zeros(l1,l2);
for tmpC1 = 1:l1
    for tmpC2 = 1:l2
        X = X1(:,:,tmpC1);
        Y = X2(:,:,tmpC2);
        
        %outS(tmpC1,tmpC2) = log(det(0.5*(X+Y))) -   0.5*log(det(X*Y)); % Comment outed by HK
        
        S = log(det(0.5*(X+Y))) -   0.5*log(det(X*Y));
        
        real_flag = isreal(S);
        if real_flag
            outS(tmpC1,tmpC2) = S;
        else
            outS(tmpC1,tmpC2) = real(S);
        end
        
        if  (outS(tmpC1,tmpC2) < 1e-10)            
            outS(tmpC1,tmpC2) = 0.0;
        end
    end
end



    