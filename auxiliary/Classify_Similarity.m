function outLabel=Classify_Similarity(Labels,x,y,mode,TestLabels,verbose)
% This code is originally created by Mehrtash Harandi (mehrtash.harandi at gmail dot com)
% This code is modified by H. Kasai for similarity-based classification.

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
    h_y_i = zeros(Number_Of_Classes,1);
    
    for tmpC2=1:Number_Of_Classes
        y_i =x(Class_Index{tmpC2},tmpC1);
        if strcmp(mode, 'linear')
            %h_y_i(tmpC2,1)=sum(abs(y_i));
            h_y_i(tmpC2,1)=sum(y_i);
        else
            h_y_i(tmpC2,1)=max(abs(y_i));
        end
    end
    [~,MinIndex]=max(h_y_i);
    outLabel(tmpC1)=MinIndex;    
    
   if verbose
        correct = (MinIndex == TestLabels(1, tmpC1));
        fprintf('# RSR: test:%03d, predict class: %03d --> ground truth :%03d (%d)\n', tmpC1, MinIndex, TestLabels(1, tmpC1), correct);
   end      
end