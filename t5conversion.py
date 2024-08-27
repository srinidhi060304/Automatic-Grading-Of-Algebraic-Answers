import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the dataset
file_path = r'D:\srinidhi\amrita\out of context\out of context\Classification_testing.xlsx'
df = pd.read_excel(file_path)

# Define the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/sentence-t5-base')

# Function to encode equations into embeddings
def encode_equation(equation):
    return model.encode(str(equation))

# Apply encoding to the "Equation" column
df['EmbeddingsLM'] = df['Equation'].apply(encode_equation)

# Create DataFrame with embeddings
t5_embeddings = pd.DataFrame(df['EmbeddingsLM'].tolist(), index=df.index).add_prefix('embed_')

# Optionally, you can save the embeddings DataFrame to a new Excel file
output_file_path = r'D:\srinidhi\amrita\out of context\out of context\testing_t5.xlsx'
t5_embeddings.to_excel(output_file_path, index=False)

# Print a message indicating completion
print("T5 embeddings have been generated and saved to:", output_file_path)
