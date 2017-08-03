% Author:
% - Mehrtash Harandi (mehrtash.harandi at gmail dot com)
%
% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.

function G = generate_Graphs(trn_X,trn_y,k_w,k_b,Metric_Flag)

nPoints = length(trn_y);
switch (Metric_Flag)
    case 1
        %AIRM metric
        tmpDist = Compute_AIRM_Metric(trn_X);
    case 2
        %Stein metric
        tmpDist = Compute_Stein_Metric(trn_X);
    otherwise
        error('The metric is not implemented.');
end %end switch


%Within Graph
G_w = zeros(nPoints);
for tmpC1 = 1:nPoints
    tmpIndex = find(trn_y == trn_y(tmpC1));
    [~,sortInx] = sort(tmpDist(tmpIndex,tmpC1));
    if (length(tmpIndex) < k_w)
        max_w = length(tmpIndex);
    else
        max_w = k_w;
    end
    G_w(tmpC1,tmpIndex(sortInx(1:max_w))) = 1;
end

%Between Graph

G_b = zeros(nPoints);
for tmpC1 = 1:nPoints
    tmpIndex = find(trn_y ~= trn_y(tmpC1));
    [~,sortInx] = sort(tmpDist(tmpIndex,tmpC1));
    if (length(tmpIndex) < k_b)
        max_b = length(tmpIndex);
    else
        max_b = k_b;
    end
    G_b(tmpC1,tmpIndex(sortInx(1:max_b))) = 1;
end

G = G_w - G_b;
end


