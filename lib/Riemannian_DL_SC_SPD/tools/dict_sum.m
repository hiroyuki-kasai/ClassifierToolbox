function S = dict_sum(B)
if iscell(B)
    S = sum(cat(3,B{:}),3);
else
    S = sum(B,2);
end
end