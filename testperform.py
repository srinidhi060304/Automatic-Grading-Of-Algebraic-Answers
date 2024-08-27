import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Excel file
file_path = "C:\\Users\\admin\\Downloads\\ML Results book.xlsx"
sheets = ['KNN', 'SVM', 'LR', 'RF', 'ADAB', 'XGB', 'DT', 'CATB', 'MLP', 'NB']

results = []

# Function to calculate metrics
def calculate_metrics(actuals, predictions):
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='weighted')
    recall = recall_score(actuals, predictions, average='weighted')
    f1 = f1_score(actuals, predictions, average='weighted')
    return accuracy, precision, recall, f1

for sheet in sheets:
    df = pd.read_excel(file_path, sheet_name=sheet)
    actuals = df['Actuals']
    
    mb_after_smote = df.iloc[:, 3]  # 4th column
    t5_after_smote = df.iloc[:, 5]  # 6th column
    
    # Calculate metrics
    mb_after_smote_metrics = calculate_metrics(actuals, mb_after_smote)
    t5_after_smote_metrics = calculate_metrics(actuals, t5_after_smote)
    
    # Append results for each metric
    results.append({
        'Model': sheet,
        'Type': 'MB after smote',
        'Accuracy': mb_after_smote_metrics[0],
        'Precision': mb_after_smote_metrics[1],
        'Recall': mb_after_smote_metrics[2],
        'F1 Score': mb_after_smote_metrics[3]
    })
    results.append({
        'Model': sheet,
        'Type': 'T5 after smote',
        'Accuracy': t5_after_smote_metrics[0],
        'Precision': t5_after_smote_metrics[1],
        'Recall': t5_after_smote_metrics[2],
        'F1 Score': t5_after_smote_metrics[3]
    })

# Create a DataFrame with the results
results_df = pd.DataFrame(results)

# Save the results to a new Excel file
results_df.to_excel('model_accuracies_after_smote.xlsx', index=False)

print("Results have been saved to 'model_accuracies_after_smote.xlsx'")
