function correct_rate = Fun_CRC(Train_M,train_label,Test_M,test_label,lambda)
% lambda is the parameter of the coefficient's regularization, e.g., lambda = 1e-3
% trainlabel is a set of numbers from 1 to nCls;

nDim     =  size(Train_M,1);
nNum     =  size(Train_M,2);
label    =  unique(train_label);
nCls     =  length(label);
Train_M  =  Train_M./( repmat(sqrt(sum(Train_M.*Train_M)), [nDim,1]) );
Test_M   =  Test_M./( repmat(sqrt(sum(Test_M.*Test_M)), [nDim,1]) );
Proj_M   =  inv(Train_M'*Train_M+lambda*eye(nNum))*Train_M';

for ti = 1:size(Test_M,2)
    y  = Test_M(:,ti);
    x  = Proj_M*y;
    for ci = 1:nCls
        class = label(ci);
        cdat = Train_M(:,train_label==class);
        er   = y - cdat*x(train_label==class);
        gap(ci) = er(:)'*er(:);
    end
    index = find(gap == min(gap));
    ID(ti) = label(index(1));
end

correct_rate = sum(ID==test_label)/length(test_label);