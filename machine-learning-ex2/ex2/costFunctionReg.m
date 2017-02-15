function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% X = 118x28 
% Theta = 28x1
% y = 118x1
% grad = 28x1
prediction = (@sigmoid(X*theta)); % 118x1
% Exclude the first theta value
theta1 = [0; theta(2:size(theta),:)];
J = (1/m)*sum((-y .* log(prediction)) - ((1-y) .* log(1-prediction))) + ((lambda/(2*m)) * sum(theta1.^2));

for i = 1:size(theta)
  if i == 1
     grad(i) = (1/m) * sum((prediction-y) .* X(:,i));
  else 
     grad(i) = ((1/m) * sum((prediction-y) .* X(:,i))) + ((lambda/m)*theta(i));
  end
end

% Vectorized

% theta1 ==> 28x1
% theta1' ==> 1x28
% theta1' x theta1 ==> 1x1 ==> goal is one number
p = lambda*(theta1'*theta1)/(2*m);

% y' = 1x118
% y' * log(prediction) ==> 1x1
% by taking the inverse of y, we get one value for J
J = ((-y)'*log(prediction) - (1-y)'*log(1-prediction))/m + p;

% Gradients
% grad = 28x1
% X' ==> 28x118
% X' * (prediction-y); 28x118 X 118x1 ==> 28x1
% lambda * theta1; 1 X 28x1 ==> 28x1
grad = (X'*(prediction - y)+lambda*theta1)/m;

% =============================================================

end
