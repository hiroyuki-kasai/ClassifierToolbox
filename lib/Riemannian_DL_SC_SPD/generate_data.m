function [train,B, A] = generate_data(n, d, num_atoms, active_size)
B = cell(num_atoms,1);
for i=1:num_atoms
    x=randn(d)-rand(d); b=x'*x/d+eye(d)*1e-2; B{i}=b/trace(b);
end
train = cell(n,1); A = zeros(num_atoms, n); 
for t=1:n
    p=rand(active_size,1);
    idx = randperm(num_atoms, active_size);
    A(idx, t) = p;
    
    X = zeros(d);
    for s=1:active_size
        X = X + p(s)*B{idx(s)};
    end
    train{t} = X;
end
end
