clear;
clc;

nClasses = 4;
nFeatures = 10;
nAugFeatures = 30;
nPoints = 1e2;
nTraining = round(2*nPoints/4);
SigmaVal =  1e-1;
noiseVal =  3e-1;
%Generate points on identity tangent space
covD_Struct.trn_X = [];
covD_Struct.trn_y = [];
covD_Struct.tst_X = [];
covD_Struct.tst_y = [];
tmpCNTR = 1;
% m = rand(nFeatures*(nFeatures+1)/2,nClasses);
m0 = repmat(rand(nFeatures*(nFeatures+1)/2,1),[1 nClasses]);
m = m0 +0.5*rand(size(m0,1),nClasses);
feature_max = zeros(nAugFeatures,1);
for tmpC1 = 1:nClasses
    tmpPoints =  repmat(m(:,tmpC1),[1,nPoints]) + SigmaVal*randn(size(m,1),nPoints);
    tmpPoints = [tmpPoints;noiseVal*randn((nAugFeatures*(nAugFeatures+1) - nFeatures*(nFeatures+1))/2,nPoints)];
    tmpPoints = Euclidean2SPD(tmpPoints);
    for tmpC2 = 1:nPoints
        tmpX = expm(tmpPoints(:,:,tmpC2));
        [V,D] = eig(tmpX);
        D = diag(D + eps);
        inv_D = 1./sqrt(D+eps);
        %         X(:,:,tmpC2) = diag(inv_D+eps)*(V*diag(D)*V');
        X(:,:,tmpC2) = (V*diag(D)*V');
        var_curr = diag(X(:,:,tmpC2));
        idx = var_curr > feature_max;
        if any(idx),feature_max(idx) = var_curr(idx); end
    end
    covD_Struct.trn_X = cat(3,covD_Struct.trn_X,X(:,:,1:nTraining));
    covD_Struct.trn_y = [covD_Struct.trn_y tmpC1*ones(1,nTraining)];
    covD_Struct.tst_X = cat(3,covD_Struct.tst_X,X(:,:,1+nTraining:end));
    covD_Struct.tst_y = [covD_Struct.tst_y tmpC1*ones(1,nPoints - nTraining)];
    
    
end
covD_Struct.nClasses = nClasses;
covD_Struct.n = nAugFeatures;
covD_Struct.r = nFeatures;
U = diag(feature_max.^(-1/2));
%Normalizing data
for tmpC1 = 1:size(covD_Struct.trn_X,3)
    covD_Struct.trn_X(:,:,tmpC1) = U*covD_Struct.trn_X(:,:,tmpC1)*U;
end
for tmpC1 = 1:size(covD_Struct.tst_X,3)
    covD_Struct.tst_X(:,:,tmpC1) = U*covD_Struct.tst_X(:,:,tmpC1)*U;
end
save('toy_data','covD_Struct');