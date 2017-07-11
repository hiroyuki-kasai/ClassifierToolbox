% takes a 3D matrix and returns a cell with length equal to the length of
% the third dimension of the matrix.
function X = lastdim2cell(Y)
X = arrayfun(@(t) Y(:,:,t), 1:size(Y,3), 'uniformoutput', false);
end