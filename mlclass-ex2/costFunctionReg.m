function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(theta'*X');h = h';
% disp(size(h));
% h2 = sigmoid((theta(2:size(theta),1))'*X(:,2:size(theta))'); 
% h2 = h2';
% J = (1/m) *(-y' *log( h ) - (1 - y)'*log(1 - h));
for i = 1:size(theta)
    if i == 1
%         theta(i,1) = theta(i,1) - alpha * (1/m)* ( h  - y)' * X(:,i);
        grad(i,1) = (1/m)* ( h  - y)' * X(:,i);
    else
%         theta(i,1) = theta(i,1) - alpha * ((1/m)* ( h  - y)' * X(:,i)' + lambda * (1/m) * theta(i,1));
        grad(i,1) = (1/m)* ( h  - y)' * X(:,i) + lambda * (1/m) * theta(i,1);
    end
end
J = (1/m) *(-y' *log( h ) - (1 - y)'*log(1 - h)) + lambda*(1/(2*m))*sum(theta(2:end).*theta(2:end)); 
% for i = 1:size(theta,1)
%     if i == 1
%         grad(1,1) = (1/m)* X'* ( h  - y);
%     else
%         grad(i,1) = (1/m)* X'* ( h  - y) + lambda * (1/m) * theta(i,1);
%     end
% end

% =============================================================

end
