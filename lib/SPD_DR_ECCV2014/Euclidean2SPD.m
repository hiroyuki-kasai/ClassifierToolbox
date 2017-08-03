function outPoints = Euclidean2SPD(inPoints)
[nFeatures,nPoints] = size(inPoints);
nFeatures = 0.5*(sqrt(1+8*nFeatures) -1);
outPoints = zeros(nFeatures,nFeatures,nPoints);
tmpSPD = ones(nFeatures);
tmpSPD(tril(tmpSPD) == 1) = 0;
tmpIdx = tmpSPD > 0;
for tmpC1 = 1:nPoints
    tmpSPD = zeros(nFeatures);
    tmpSPD(tmpIdx) = inPoints(nFeatures+1:end,tmpC1)./sqrt(2);
    tmpSPD = tmpSPD + tmpSPD';
    tmpSPD(1:nFeatures+1:end) = inPoints(1:nFeatures,tmpC1);
    outPoints(:,:,tmpC1) = tmpSPD;    
end %end tmpC1
