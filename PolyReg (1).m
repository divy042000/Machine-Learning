% Generate synthetic data
rng(42); % Set random seed for reproducibility
X = 2 * rand(100, 1);
y = 4 + 3 * X + randn(100, 1);

% Split the data into training and test sets
rng(42); % Reset random seed for consistent split
idx = randperm(length(X));
X_train = X(idx(1:80));
y_train = y(idx(1:80));
X_test = X(idx(81:end));
y_test = y(idx(81:end));

% Linear Regression Model
linear_model = fitlm(X_train, y_train);

% Plot linear regression
figure;
scatter(X, y, 'blue', 'DisplayName', 'Data points');
hold on;
plot(X_train, predict(linear_model, X_train), 'red', 'DisplayName', 'Linear Regression');
title('Linear Regression');
xlabel('X');
ylabel('y');
legend('show');

% Polynomial Regression Model (degree = 15)
p = polyfit(X_train, y_train, 15);
y_poly_train = polyval(p, X_train);
y_poly_test = polyval(p, X_test);

% Plot polynomial regression
figure;
scatter(X, y, 'blue', 'DisplayName', 'Data points');
hold on;
plot(X_train, y_poly_train, 'red', 'DisplayName', 'Polynomial Regression');
title('Polynomial Regression (Degree=15)');
xlabel('X');
ylabel('y');
legend('show');

% Evaluate models
mse_train_linear = mean((y_train - predict(linear_model, X_train)).^2);
mse_test_linear = mean((y_test - predict(linear_model, X_test)).^2);

fprintf('Linear Regression - Training MSE: %.2f\n', mse_train_linear);
fprintf('Linear Regression - Test MSE: %.2f\n', mse_test_linear);

mse_train_poly = mean((y_train - y_poly_train).^2);
mse_test_poly = mean((y_test - y_poly_test).^2);

fprintf('Polynomial Regression (Degree=15) - Training MSE: %.2f\n', mse_train_poly);
fprintf('Polynomial Regression (Degree=15) - Test MSE: %.2f\n', mse_test_poly);

%% % Generate synthetic data
clear;
close all;
%rng(42); % Set random seed for reproducibility
X = 2 * rand(100, 1);
y = 4 + 3 * X + randn(100, 1);

% Split the data into training and test sets
%rng(42); % Reset random seed for consistent split
idx = randperm(length(X));
X_train = X(idx(1:80));
y_train = y(idx(1:80));
X_test = X(idx(81:end));
y_test = y(idx(81:end));

% Reshape X_train for ridge regression
X_train = [ones(length(X_train), 1), X_train]; % Add a column of ones for the intercept
X_test = [ones(length(X_test), 1), X_test]; % Add a column of ones for the intercept

% Ridge Regression Model (L2 regularization)
lambda = 100; % Regularization parameter (adjust as needed)

% Calculate ridge coefficients using the normal equation
beta = (X_train' * X_train + lambda * eye(size(X_train, 2))) \ (X_train' * y_train);

% Plot ridge regression
figure;
scatter(X, y, 'blue', 'DisplayName', 'Data points');
hold on;
plot(X_train(:, 2), X_train * beta, 'red', 'DisplayName', 'Ridge Regression');
title('Ridge Regression');
xlabel('X');
ylabel('y');
legend('show');

% Evaluate ridge regression model
y_train_pred_ridge = X_train * beta;
y_test_pred_ridge = X_test * beta;

mse_train_ridge = mean((y_train - y_train_pred_ridge).^2);
mse_test_ridge = mean((y_test - y_test_pred_ridge).^2);

fprintf('Ridge Regression - Training MSE: %.2f\n', mse_train_ridge);
fprintf('Ridge Regression - Test MSE: %.2f\n', mse_test_ridge);

