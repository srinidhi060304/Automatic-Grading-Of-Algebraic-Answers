This folder contains 5 folders and some programs and excels.
The folders include
1.Database separated
2.MB_after_SMOTE
3.MB_before_SMOTE
4.T5_afterSMOTE
5.T5_beforeSMOTE

1. Database Separated: Contains 10 excel sheets with around 50 answers and the marks and the classification for each.
In classification:
1 corresponds to correct,2 corresponds to partially correct, 3 corresponds to incorrect, 4 to out of context and 5 to irrelevent.
In marks:
marks are alloted 1-5 as nessessary.

2.MB_after_SMOTE: Contains 10 programs and 10 excel sheets. The 10 programs contain training and texting codes for 
the database after SMOTE is applied using different models. The 10 excel sheets hold the predicted classification for each question of the test set.
Here the database is for a mathbert dataset.

3.MB_before_SMOTE: Contains 10 programs and 10 excel sheets. The 10 programs contain training and texting codes for 
the database before SMOTE is applied using different models. The 10 excel sheets hold the predicted classification for each question of the test set.
Here the database is for a mathbert dataset.

4.T5_after_SMOTE: Contains 10 programs and 10 excel sheets. The 10 programs contain training and texting codes for 
the database after SMOTE is applied using different models. The 10 excel sheets hold the predicted classification for each question of the test set.
Here the database is for a T5 dataset.

5.T5_before_SMOTE: Contains 10 programs and 10 excel sheets. The 10 programs contain training and texting codes for 
the database before SMOTE is applied using different models. The 10 excel sheets hold the predicted classification for each question of the test set.
Here the database is for a T5 dataset.

ALL COMBINED.xlsx contains the same data as database_separated but all under one excel sheet for better understanding.

ML Results book.xlsx contains the combined predictions of all models both MathBERT and T5 for better understanding and comparison.

ML_RESULTS-1.xlsx contains all the results of the training data( in terms of the performance matrices).

model_accuracies.xlsx contains all the model accuracies for the text set.

model_accuracies_after_SMOTE.xlsx contains the model accuracies of the test set only after SMOTE has been applied.

resampled_t5_embeddings.xlsx contains all training t5 embeddings after SMOTE is applied.

resampled_training_mathbert.xlsx contains all the training mathbert embeddings after SMOTE is applied.

SMOTE_analysis.py contains code to apply SMOTE onto an excel file.

t5_embeddings.xlsx contains all the T5 embeddings before SMOTE is applied.

t5conversion.py contains code to convert an excel sheet into T5 embeddings.

mathbertconversion.py contains code to convert an excel sheet into mathbert embeddings.

testing_mathbert.xlsx contains mathbert form of the test set.

testing_t5.xlsx contains t5 embeddings of the test set.

testperform.py contains code to find performance matrices for the test set from the excel sheet.

training_mathbert.xlsx contains all the mathBERT embedding of the training dataset.