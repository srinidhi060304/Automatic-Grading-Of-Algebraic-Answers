import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.svm import SVC

# Load the dataset
file_path = r'D:\srinidhi\amrita\out of context\out of context\training_mathbert.xlsx'
df = pd.read_excel(file_path)

# Separate features and target
X_train = df.iloc[:, :-1]  # Features (embed_0 to embed_383)
y_train = df['Classification']  # Target

# Initialize SVM classifier
svm = SVC(kernel='linear')

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameters for grid search
param_grid = {'C': [0.1, 1, 10, 100]}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=kf, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_score)

# Now, you can use the best parameters found by grid search for SVM
best_svm = SVC(kernel='linear', C=best_params['C'])

# Perform cross-validation and calculate evaluation metrics
accuracy = cross_val_score(best_svm, X_train, y_train, cv=kf, scoring='accuracy')

# Print the training results
print("\nTraining Results:")
print("Accuracy mean: {:.2f}".format(accuracy.mean()))
print("Accuracy std: {:.2f}".format(accuracy.std()))

# Load testing data
testing_data = pd.read_excel(r'D:\srinidhi\amrita\out of context\out of context\testing_mathbert.xlsx')

# Assuming you already have 'X_test' containing the features (embeddings) for testing
X_test = testing_data  # No need to exclude the last column

# Predict classifications for testing data
predictions = grid_search.predict(X_test)

# Add predictions to the testing data DataFrame
testing_data['Predicted_Classification'] = predictions

# Save the testing data with predicted classifications to a new Excel file
testing_output_file = "predicted_testing_mathbert_svm.xlsx"
testing_data.to_excel(testing_output_file, index=False)
print("Predictions saved to:", testing_output_file)
