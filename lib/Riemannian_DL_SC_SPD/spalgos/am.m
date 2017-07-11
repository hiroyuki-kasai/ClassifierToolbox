function a = am(A,w)
   if nargin < 2
      w=ones(numel(A),1);
      w=w/length(w);
   end
   
   a = w(1)*A{1};
   for i=2:numel(A)
      a = a + w(i)*A{i};
   end
end