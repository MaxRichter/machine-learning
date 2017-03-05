function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X = 12x2 (first column are all 1)
% y = 12x1
% theta = 2x1; 1;1
% theta1 = 2x1
% lambda = 1

theta1 = [0; theta(2:size(theta),:)];
prediction = X*theta; % X=12x2 theta=2x1 ==> 12x1
error = (prediction-y); % ==> 12x1
regularization = lambda*(theta1'*theta1)/(2*m);

J = (error'*error)/(2*m) + regularization;

% Gradient
% grad = 2x1
% X = 12x2
% X' = 2x12
% X'*error = 2x1
% lambda*theta1 = 2x1

grad = (X'*error + lambda*theta1)/m;

% =========================================================================

grad = grad(:);

end
