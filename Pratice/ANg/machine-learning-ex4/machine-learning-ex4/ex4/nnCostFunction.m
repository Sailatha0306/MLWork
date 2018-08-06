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
n = size(X,2);
         
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
X=[ones(m,1) X];

A2=sigmoid(Theta1*X');
A2=[ones(1,m);A2];

htheta=sigmoid(Theta2*A2);

y_row=size(y,1);
final_y=zeros(num_labels,size(y,1));

for i=1:y_row
 final_y(y(i,1),i)=1;
end
jtheta=0;

  jtheta = sum(sum((-final_y.*log(htheta)).-((1.-final_y)).*log(1.-htheta)));

J=jtheta/m;

nthetha1=Theta1(:,2:size(Theta1,2));
nthetha2=Theta2(:,2:size(Theta2,2));

J=J+((lambda/(2*m))*((sum(sum(nthetha1.^2)))+(sum(sum(nthetha2.^2)))));

%Theta1_grad = Theta1_grad(:,2:end);
%Theta2_grad = Theta2_grad(:,2:end);
for i =1:m
  a1=X(i,:);%1X401
  z2=Theta1*a1';%25X401X401X1
  %z2=[1;z2];
  a2=sigmoid(z2);%25 X 1
  a2=[1 a2'];%1X26
  a3=sigmoid(Theta2*a2');%10 X 26 * 26 X 1-->10 X 1 
  
  epi3=a3.-final_y(:,i);%10 X 1
<<<<<<< HEAD
  epi2=(Theta2'*epi3).*sigmoidGradient(z2);%26 X 1
  a2=a2(2:end);
  delta2=epi3*a2';%10 X 1 * 1 X 25= 10 X 25
=======
  epi2=(Theta2(:,2:end)'*epi3).*sigmoidGradient(z2);%25 X 1
  %a2=a2(2:end);
  Theta2_grad = Theta2_grad + epi3*a2;%10 X 1 * 1 X 25= 10 X 25
  Theta1_grad = Theta1_grad + epi2*a1;%25 X 1 * 1 X 400= 25 X 400
  %delta2=delta2 + epi3*a2';%10 X 1 * 1 X 25= 10 X 25
>>>>>>> e22362b4673c3ea17b3ff73602d5582ff509296c
end
  Theta2_grad(:,1)=(Theta2_grad(:,1))./m;
  Theta1_grad(:,1)=(Theta1_grad(:,1))./m;
  Theta2_grad(:,2:end)=((Theta2_grad(:,2:end))./m)+(Theta2(:,2:end)).*(lambda/m);
  Theta1_grad(:,2:end)=((Theta1_grad(:,2:end))./m)+(Theta1(:,2:end)).*(lambda/m);
  %%delta1=delta1+epi2*a1';%26 X 1 * 1 X 401= 
% -------------------------------------------------------------

% =========================================================================
%size(Theta2_grad)
%size(Theta1_grad)
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%size(grad)

end
