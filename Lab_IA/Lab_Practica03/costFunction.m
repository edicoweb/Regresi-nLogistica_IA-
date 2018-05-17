function [J, grad] = costFunction(theta, X, y)
m = length(y);  
J = 0;
grad = zeros(size(theta));
z = X * theta;
g = sigmoid(z);
sigma1 = -(y .* log(g)) - ((1 - y) .* log(1 - g));
J = (1 / m) * sum(sigma1);

sigma2 = (g - y) .* X ;
grad = (1 / m) * sum(sigma2);
end
