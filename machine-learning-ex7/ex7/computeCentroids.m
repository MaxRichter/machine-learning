function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% X = 300x2
% idx = 300x1
% centroids = 3x2
% K = 1x1 = 3
% n = 1x1 = 2
% m = 1x1 = 300

% Loop over centroids
for i = 1:K
  % For every K, match the idx with the coordinates from X
  X_i = X(idx == i, :); % nx2 (x,y coordinates)
  % Update the corresponding centroid using octave mean function
  % size(mean(X_i)) = 1x2 so that it fits to centroids(i, :) 1x2
  centroids(i, :) = mean(X_i);
end






% =============================================================


end

