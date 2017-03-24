function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% X = 50x2
% K = 1
% U = 2x2
% Z = 50x1

U_reduce = U(:, 1:K); % 2x1
for i = 1: size(X, 1)
  Z(i, :) = (U_reduce'*X(i, :)'); % 1x2 (U_reduce') X 2x1 (X(i, :)') ==> 1x1
end

%U_reduce = U(:, 1:K); % 2x1
%Z = X * U_reduce; % 50x2 X 2x1 ==> 50x1

% =============================================================

end
