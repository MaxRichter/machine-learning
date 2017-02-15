function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); % 3x1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% X = 100x3 
% Theta = 3x1
% y = 100x1

prediction = (@sigmoid(X*theta)); % 100x1
J = (1/m)*sum((-y .* log(prediction)) - ((1-y) .* log(1-prediction)));

grad(1) = (1/m) * sum((prediction-y) .* X(:,1)); % (predction - y) ==> 100x1 X(:,1) ==> 100x1
grad(2) = (1/m) * sum((prediction-y) .* X(:,2));
grad(3) = (1/m) * sum((prediction-y) .* X(:,3));

% Vectorized

% Cost
% y' = 1x100
% y * prediction = 100x1 X 100x1 ==> 100x1 // Not what we search for
% y' * predction = 1x100 X 100x1 ==> 1x1 // This is what we look for

J = ((-y)' * log(prediction) - (1-y)' * log(1-prediction))/m;

% Gradient ==> 3x1
% prediction ==> 100x1
% X = 100x3
% X' = 3x100
% X' * prediction; 3x100 X 100x1 ==> 3x1
grad = (X' * (prediction-y))/m;

% =============================================================

end
