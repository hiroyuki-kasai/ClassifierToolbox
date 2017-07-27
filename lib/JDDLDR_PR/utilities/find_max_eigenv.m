
%----------------------------------------------------------------------
%Find the largest eigenvalue
%----------------------------------------------------------------------

function [ opts ] = find_max_eigenv ( matrix )

[v,s]     =       eig(matrix);
sort_s    =       sort(diag(s),'descend');
opts      =       sort_s(1);