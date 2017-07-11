function X=spd(d)
X = randn(d) - rand(d);
X = X'*X;
end