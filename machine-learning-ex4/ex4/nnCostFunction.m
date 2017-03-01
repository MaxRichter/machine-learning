function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% 3-layer neural network
% input_layer_size = 400
% hidden_layer_size = 25
% num_labels = 10
% m = 5000
% X = 5000x400
% y = 5000x1
% lambda = 0
% Theta1 = 25x401
% Theta2 = 10x26

% Add ones to the first column of data matrix X
a1 = [ones(m, 1) X]; % 5000x401
% Calculate the hidden layer
z2 = Theta1 * a1'; % 25x5000
% Add ones to the hidden layer in the first column
a2 = [ones(1, size(z2, 2)); @sigmoid(z2)]; % 26x5000
% Calculate the output layer
z3 = Theta2 * a2; % 10x5000 
a3 = @sigmoid(z3); % 10x5000
h = a3; % 10x5000

% Now we tranform the y result vector into a matrix where 1s in the
% columns map to the corresponding values of y
yMatrix = zeros(num_labels, m);

for i=1:num_labels,
    yMatrix(i,:) = (y==i); % 10x5000
end

% Now that we have y as a 10x5000 matrix instead of a 5000x1 vector,
% we can use it to calculate our cost as compared to h (which is a3)

% Note that for this vectorized implementation, y(i)k is given as
% yMatrix and h is given as h(thetha)(x(i))k

J = (sum( sum( -1*yMatrix.*log(h) - (1 - yMatrix).*log(1-h) ) ))/m;

% Implementing regularization

% For this we can steal some of the logic from ex2 costFunctionReg.m
% First, we toss the first columns of each Theta(i) matrix.
Theta1Reg = Theta1(:,2:size(Theta1,2));
Theta2Reg = Theta2(:,2:size(Theta2,2));

% Now implement the regularization formula described on page 6 of ex4.
Reg = (lambda/(2*m)) * (sum(sum( Theta1Reg.^2 )) + sum( sum( Theta2Reg.^2 ) ));

% Now just add the regularization term to the previously calculated J
J = J + Reg;


% Vectorized
% calculte penalty
% p = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));
% calculate J
% J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m + lambda*p/(2*m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
X = [ones(m,1) X];

for t = 1:m
  % Step 1 - Feedforward
  a1 = X(t,:); % 1x401 % Get velues form column m
  z2 = Theta1 * a1'; % 25x1
  a2 = [ones(1, size(z2, 2)); @sigmoid(z2)]; % 26x1
  z3 = Theta2 * a2; % 10x1 
  a3 = @sigmoid(z3); % 10x1
  
  % Step 2 - Calculate d3 from the output layer
  d3 = a3 - yMatrix(:,t); % Get only the values from column m % 10x1
  % Step 3 - Calculate d3 for the hidden layer L2
  % Re-add a bias node for z2
  z2 = [1; z2]; % 26x1
  d2 = (Theta2' * d3) .* sigmoidGradient(z2); % 26x1
  % Step 4 - Strip out bias node from resulting d2
  d2 = d2(2:end); % 25x1
  Theta2_grad = (Theta2_grad + d3 * a2'); % 10x26
  Theta1_grad = (Theta1_grad + d2 * a1); % 25x401
  
end

% Step 5 - Calculate the (unregulized) gradients
% Note that for regularization these will have to removed/commented out.
%Theta2_grad = Theta2_grad ./ m;
%Theta1_grad = Theta1_grad ./ m;  

% Part 3: Implement regularization with the cost function and gradients.

% Implement for Theta1 and Theta2 when l = 0
Theta1_grad(:,1) = Theta1_grad(:,1)./m;
Theta2_grad(:,1) = Theta2_grad(:,1)./m;

% Implement for Theta1 and Theta 2 when l > 0
Theta1_grad(:,2:end) = Theta1_grad(:,2:end)./m + ( (lambda/m) * Theta1(:,2:end) );
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)./m + ( (lambda/m) * Theta2(:,2:end) );

% Vectorized
% calculate sigmas
% sigma3 = a3.-Y;
% sigma2 = (sigma3*Theta2).*sigmoidGradient([ones(size(z2, 1), 1) z2]);
% sigma2 = sigma2(:, 2:end);

% accumulate gradients
% delta_1 = (sigma2'*a1);
% delta_2 = (sigma3'*a2);

% calculate regularized gradient
% p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
% p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
% Theta1_grad = delta_1./m + p1;
% Theta2_grad = delta_2./m + p2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
