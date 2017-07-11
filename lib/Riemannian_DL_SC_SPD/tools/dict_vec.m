function B = dict_vec(BB)
if size(BB,2) ~= 1
    BB = BB';
end
B = cell2mat(cellfun(@(x) vec(x), BB, 'uniformoutput', false)');
end