# Muhammad Athallah Yakarazi
# 24/532752/PA/22532

import math

import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# load data and define column names
file_name = 'M_Athallah_Y_SLP_fin.xlsx-Data.csv'
column_names = ['X1', 'X2', 'X3', 'X4', 'species']
dataset = pd.read_csv(file_name, header=None, names=column_names)

X = dataset.drop(columns=['species'])
y = dataset['species']

# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = .10)
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.2, stratify = y)

# verify split
print('Training features shape:', X_train.shape)
print('Validation features shape:', X_val.shape)
print('Training target shape:', Y_train.shape)
print('Validation target shape:', Y_val.shape)

Y_train_num = Y_train.apply(lambda x: 0 if x == 'Iris-setosa' else 1) # convert name to number for calculation
Y_val_num = Y_val.apply(lambda x: 0 if x == 'Iris-setosa' else 1)

learning_rate = 0.1

bias = 0.5
teta1 = 0.5
teta2 = 0.5
teta3 = 0.5
teta4 = 0.5

target_epochs = 5

# list of bias and teta of each epoch to be used for validation later
history_bias = []
history_teta1 = []
history_teta2 = []
history_teta3 = []
history_teta4 = []

train_accuracies = []
val_accuracies = []
train_loss = []
val_loss = []

# training
X_train_vals = X_train.values
Y_train_vals = Y_train_num.values
n_samples = len(X_train_vals)

for epoch in range(target_epochs):
    total_sse = 0 # track total error per epoch
    correct_predictions = 0
    
    for i in range(n_samples):
        X1, X2, X3, X4 = X_train_vals[i]
        target = Y_train_vals[i]
        
        dot_z = bias + (teta1 * X1) + (teta2 * X2) + (teta3 * X3) + (teta4 * X4)
        sigmoid_z = 1 / (1 + math.exp(-dot_z))
        
        prediction = 1 if sigmoid_z > 0.5 else 0
        if prediction == target:
            correct_predictions += 1
        
        error = sigmoid_z - target
        sum_square_error = error*error
        total_sse += sum_square_error
        
        common_grad = 2 * (sigmoid_z - target) * (1 - sigmoid_z) * sigmoid_z
        dbias = common_grad
        dteta1 = common_grad * X1
        dteta2 = common_grad * X2
        dteta3 = common_grad * X3
        dteta4 = common_grad * X4
        
        bias -= learning_rate * dbias
        teta1 -= learning_rate * dteta1
        teta2 -= learning_rate * dteta2
        teta3 -= learning_rate * dteta3
        teta4 -= learning_rate * dteta4
    
    history_bias.append(bias)
    history_teta1.append(teta1)
    history_teta2.append(teta2)
    history_teta3.append(teta3)
    history_teta4.append(teta4)
    
    mse = total_sse / n_samples
    accuracy = correct_predictions / n_samples
    train_accuracies.append(accuracy)
    train_loss.append(mse)
    
    print(f'Epoch {epoch + 1}/{target_epochs} - MSE: {mse:.4f}, Accuracy: {accuracy:.4f}')

# validation
X_val_vals = X_val.values
Y_val_vals = Y_val_num.values
n_samples_val = len(X_val_vals)

for epoch in range(target_epochs):
    total_sse = 0
    correct_predictions = 0
    
    bias = history_bias[epoch]
    teta1 = history_teta1[epoch]
    teta2 = history_teta2[epoch]
    teta3 = history_teta3[epoch]
    teta4 = history_teta4[epoch]
    
    for i in range(n_samples_val):
        X1, X2, X3, X4 = X_val_vals[i]
        target = Y_val_vals[i]
        
        dot_z = bias + (teta1 * X1) + (teta2 * X2) + (teta3 * X3) + (teta4 * X4)
        sigmoid_z = 1 / (1 + math.exp(-dot_z))
        
        prediction = 1 if sigmoid_z > 0.5 else 0
        if prediction == target:
            correct_predictions += 1
        
        error = sigmoid_z - target
        sum_square_error = error*error
        total_sse += sum_square_error
        
    mse_val = total_sse / n_samples_val
    accuracy_val = correct_predictions / n_samples_val
    val_accuracies.append(accuracy_val)
    val_loss.append(mse_val)
    
    print(f'Validation Epoch {epoch + 1}/{target_epochs} - MSE: {mse_val:.4f}, Accuracy: {accuracy_val:.4f}')
    
# chart
epochs_range = range(1, target_epochs + 1)

plt.figure(figsize=(8, 5))
#plt.plot(epochs_range, train_accuracies, marker='o', linestyle='-', color='blue', label='Training Accuracy')
#plt.plot(epochs_range, val_accuracies, marker='s', linestyle='-', color='red', label='Validation Accuracy')
plt.plot(epochs_range, train_loss, marker='o', linestyle='--', color='blue', label='Training Loss')
plt.plot(epochs_range, val_loss, marker='s', linestyle='--', color='red', label='Validation Loss')
'''
plt.title('Training and Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs_range)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''
plt.title('Training and Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs_range)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()