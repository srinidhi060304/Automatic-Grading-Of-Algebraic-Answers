import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = r'D:\srinidhi\amrita\out of context\out of context\t5_embeddings.xlsx'
df = pd.read_excel(file_path)

# Separate features and target
X = df.iloc[:, :-1]  # Features (embed_0 to embed_383)
y = df['Classification']  # Target

# Initialize Gaussian Naive Bayes classifier
naive_bayes = GaussianNB()

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameters for grid search
param_grid = {}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=naive_bayes, param_grid=param_grid, cv=kf, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X, y)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_score)

# Now, you can use the best parameters found by grid search for Gaussian Naive Bayes
best_naive_bayes = GaussianNB(**best_params)

# Perform cross-validation and calculate evaluation metrics
accuracy = cross_val_score(best_naive_bayes, X, y, cv=kf, scoring='accuracy')
precision = cross_val_score(best_naive_bayes, X, y, cv=kf, scoring='precision_macro')
recall = cross_val_score(best_naive_bayes, X, y, cv=kf, scoring='recall_macro')
f1 = cross_val_score(best_naive_bayes, X, y, cv=kf, scoring='f1_macro')

# Print the results
print("\nPrecision:")
print("  - Mean: {:.2f}".format(precision.mean()))
print("  - Standard Deviation: {:.2f}".format(precision.std()))

print("\nRecall:")
print("  - Mean: {:.2f}".format(recall.mean()))
print("  - Standard Deviation: {:.2f}".format(recall.std()))

print("\nF1 Score:")
print("  - Mean: {:.2f}".format(f1.mean()))
print("  - Standard Deviation: {:.2f}".format(f1.std()))

print("\nAccuracy:")
print("  - Mean: {:.2f}".format(accuracy.mean()))
print("  - Standard Deviation: {:.2f}".format(accuracy.std()))

# Load testing data
testing_data = pd.read_excel(r'D:\srinidhi\amrita\out of context\out of context\testing_t5.xlsx')

# Assuming you already have 'X_test' containing the features (embeddings) for testing
X_test = testing_data  # No need to exclude the last column

best_naive_bayes.fit(X,y)

# Predict classifications for testing data
predictions = best_naive_bayes.predict(X_test)

# Add predictions to the testing data DataFrame
testing_data['Predicted_Classification'] = predictions

# Save the testing data with predicted classifications to a new Excel file
testing_output_file = "predicted_testing_mathbert_naive_bayes.xlsx"
testing_data.to_excel(testing_output_file, index=False)
print("Predictions saved to:", testing_output_file)
