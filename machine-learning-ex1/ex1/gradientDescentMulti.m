function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_vector = [0; 0; 0];
X_zero = X(:,1); % 47x1
X_one = X(:,2); % 47x1
X_two = X(:,3); % 47x1

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    %prediction = X*theta; % 47x2 X 2x1 ==> 47x1
    %distance = prediction - y; % 47x1 - 47x1  
    
    %theta_zero = theta(1) - alpha*(1/m)*sum(distance .* X_zero); % elementwise multiplication because of the i-index
    %theta_one = theta(2) - alpha*(1/m)*sum(distance .* X_one);
    %theta_two = theta(3) - alpha*(1/m)*sum(distance .* X_two);
    
    %theta = [theta_zero; theta_one; theta_two]; % write the updated values back to theta

    %theta_vector = theta_vector - alpha*(1/m) * X' * (X * theta_vector - y) ;
    theta = theta - alpha*(1/m) * X' * (X * theta - y) ;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
