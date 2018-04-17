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

htheta = sigmoid(theta'*X');
n=size(theta,1);
J = (sum((-y'.*log(htheta)).-((1.-y').*log(1.-htheta)))/m)+((sum(theta([2,n],:).^2)).*(lambda./(2.*m))) ;

for i = 1:size(theta,1)
  if i == 1
    grad(i,:)=(sum((htheta.-y').*X'(i,:))./m);1
  else
    grad(i,:)=(sum((htheta.-y').*X'(i,:))./m)+((lambda./m).*theta(i,:));
   end
end 




% =============================================================

end
