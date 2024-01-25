clear
close all
% Gradient Descent Example for a Quadratic Function

% Define the quadratic function: f(x) = x^2 + 4x + 4
f = @(x) x.^2;

% Define the gradient of the function: f'(x) = 2x + 4
grad_f = @(x) 2.*x;

% Set the learning rate and number of iterations
learning_rate = 0.005;
num_iterations = 1000;

% Initialize the starting point
x_init = 4;

% Initialize an array to store the values of x during each iteration
x_values = zeros(1, num_iterations+1);
x_values(1) = x_init;
tic
% Gradient Descent Iterations
for i = 1:num_iterations
    % Update x using the gradient descent update rule
    x_new = 0.99*x_values(i) - learning_rate * grad_f(x_values(i));
    
    % Store the updated x value
    x_values(i+1) = x_new;
end
elapsed_time=toc;
% Display the results
fprintf('Initial x: %f\n', x_init);
fprintf('Minimum x (approx): %f\n', x_values(end));
fprintf('Minimum value of the function: %f\n', f(x_values(end)));
fprintf('Time taken by gradient descent: %.6f seconds\n', elapsed_time);


% Plot the function and the path taken by gradient descent
figure;
x_range = linspace(-5, 5, 100);
y_values = f(x_range);
plot(x_range, y_values, 'LineWidth', 2);
hold on;
scatter(x_values, f(x_values), 'r', 'filled');
title('Gradient Descent for Quadratic Function');
xlabel('x');
ylabel('f(x)');
legend('Quadratic Function', 'Gradient Descent Path', 'Location', 'best');
grid on;
hold off;
