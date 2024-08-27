import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = r'D:\srinidhi\amrita\out of context\out of context\training_mathbert.xlsx'
df = pd.read_excel(file_path)

# Separate features and target
X_train = df.iloc[:, :-1]  # Features (embed_0 to embed_383)
y_train = df['Classification']  # Target

# Initialize Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_score)

# Now, you can use the best parameters found by grid search for Random Forest
best_rf = RandomForestClassifier(**best_params, random_state=42)

# Perform cross-validation and calculate evaluation metrics
accuracy = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring='accuracy')
precision = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring='precision_macro')
recall = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring='recall_macro')
f1 = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring='f1_macro')

# Print the training results
print("\nTraining Results:")
print("Precision mean: {:.2f}".format(precision.mean()))
print("Precision std: {:.2f}".format(precision.std()))
print("Recall mean: {:.2f}".format(recall.mean()))
print("Recall std: {:.2f}".format(recall.std()))
print("F1 Score mean: {:.2f}".format(f1.mean()))
print("F1 Score std: {:.2f}".format(f1.std()))
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
testing_output_file = "predicted_testing_mathbert_rf.xlsx"
testing_data.to_excel(testing_output_file, index=False)
print("Predictions saved to:", testing_output_file)
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = r'D:\srinidhi\amrita\out of context\out of context\training_mathbert.xlsx'
df = pd.read_excel(file_path)

# Separate features and target
X_train = df.iloc[:, :-1]  # Features (embed_0 to embed_383)
y_train = df['Classification']  # Target

# Initialize Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_score)

# Now, you can use the best parameters found by grid search for Random Forest
best_rf = RandomForestClassifier(**best_params, random_state=42)

# Perform cross-validation and calculate evaluation metrics
accuracy = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring='accuracy')
precision = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring='precision_macro')
recall = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring='recall_macro')
f1 = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring='f1_macro')

# Print the training results
print("\nTraining Results:")
print("Precision mean: {:.2f}".format(precision.mean()))
print("Precision std: {:.2f}".format(precision.std()))
print("Recall mean: {:.2f}".format(recall.mean()))
print("Recall std: {:.2f}".format(recall.std()))
print("F1 Score mean: {:.2f}".format(f1.mean()))
print("F1 Score std: {:.2f}".format(f1.std()))
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
testing_output_file = "predicted_testing_mathbert_rf.xlsx"
testing_data.to_excel(testing_output_file, index=False)
print("Predictions saved to:", testing_output_file)
