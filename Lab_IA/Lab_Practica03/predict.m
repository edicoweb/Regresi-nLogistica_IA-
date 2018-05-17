function p = predict(theta, X)
m = size(X, 1); 
p = zeros(m, 1);
d = sigmoid(X * theta);
for i = 1:m
	if(d(i,1) >= 0.5)
		p(i,1) = 1;
	endif
endfor
end
