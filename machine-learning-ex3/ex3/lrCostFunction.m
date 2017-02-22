function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

size(X);
size(theta);
size(y);
size(lambda);
size(grad);
% X = 5x4 
% Theta = 4x1
% y = 5x1
% grad = 4x1
% Lambda = 1x1

prediction = (@sigmoid(X*theta)); % 5x1
% Exclude the first theta value
theta1 = [0; theta(2:size(theta),:)];

% Vectorized

% theta1 ==> 4x1
% theta1' ==> 1x4
% theta1' x theta1 ==> 1x1 ==> goal is one number
% Create the penalty term
p = lambda*(theta1'*theta1)/(2*m);

% y' = 1x5
% y' * log(prediction) ==> 1x1
% By taking the inverse of y, we get one value for J
J = ((-y)'*log(prediction) - (1-y)'*log(1-prediction))/m + p;

% Gradients
% grad = 4x1
% X' ==> 4x5
% X' * (prediction-y); 4x5 X 5x1 ==> 4x1
% lambda * theta1; 1 X 4x1 ==> 4x1
grad = (X'*(prediction - y)+lambda*theta1)/m;

% =============================================================

grad = grad(:);

end
