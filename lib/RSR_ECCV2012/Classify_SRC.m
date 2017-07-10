% Author:
% - Mehrtash Harandi (mehrtash.harandi at gmail dot com)
%
% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.

function outLabel = Classify_SRC(A,Labels,x,y,TestLabels,verbose)
%x: sparse solution
%y: original query

[~,Number_Query]=size(y);
%Indices
Number_Of_Classes=max(Labels);
Class_Index = cell(Number_Of_Classes);
for tmpC1=1:Number_Of_Classes
    Class_Index{tmpC1}=find(Labels==tmpC1);
end

outLabel = zeros(1,Number_Query);
for tmpC1=1:Number_Query
    res_y=zeros(Number_Of_Classes,1);
    for tmpC2=1:Number_Of_Classes
        delta_x=zeros(size(x,1),1);
        delta_x(Class_Index{tmpC2})=x(Class_Index{tmpC2},tmpC1);
        res_y(tmpC2,1)=norm(y(:,tmpC1)-A*delta_x);
    end
    [~,MinIndex]=min(res_y);
    outLabel(tmpC1)=MinIndex; 
    
   if verbose
        correct = (MinIndex == TestLabels(1, tmpC1));
        fprintf('# RSR: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', tmpC1, MinIndex, TestLabels(1, tmpC1), correct);
   end    
end