function r = randperm(a,b)
if length(a)==1
    x = 1:a; 
elseif length(a)>1
    x=a;
end    
if nargin == 1
    b = length(x);
end
if length(x)<=b
    b = length(x);
    fprintf(['Warning: a=' num2str(a) ' b=' num2str(b) '\n']);
end

r = zeros(1,b);
for i=1:b
    pick = ceil(rand*length(x));
    r(i) = x(pick);
    x(pick) = [];
end
end