function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_vector = [0; 0];
X_zero = X(:,1); % 97x1
X_one = X(:,2); % 97x1

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    prediction = X*theta; % 97x2 X 2x1 ==> 97x1
    distance = prediction - y; % 97x1 - 97x1  
    
    theta_zero = theta(1) - alpha*(1/m)*sum(distance .* X_zero); % elementwise multiplication because of the i-index
    theta_one = theta(2) - alpha*(1/m)*sum(distance .* X_one);
    
    theta = [theta_zero; theta_one]; % write the updated values back to theta
    
    theta_vector = theta_vector - alpha*(1/m) * X' * (X * theta_vector - y); % X' ==> 2x97 | distance ==> 97x1 ==> results in: 2x1 
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
