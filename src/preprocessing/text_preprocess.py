import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Function for decamelcasing, handles camelCase and PascalCase
# Add space before uppercase letters (except at the start)
def decamelcase(text):
    text = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', text)
    return text

# Function to remove specific template phrases from bug descriptions
# Use regex to find and remove the specific phrases, ignoring case
# Removes the phrase itself, colon (optional), and leading whitespace
def remove_template_phrases(text):
    pattern = re.compile(
        r'(Detailed problem description:|Step by step, how to reproduce:|Available workarounds:)\s*:?\s*',
        re.IGNORECASE
    )
    return pattern.sub('', text)

# Main text preprocessing function
def preprocess_text(text, stop_words, lemmatizer):
    
    # Handle potential non-string data like NaN or None
    if not isinstance(text, str):
        return "" 

    # Decamelcase first, as it creates new words
    text = decamelcase(text)

    # Case folding
    text = text.lower()

    # Remove punctuation and numbers
    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Stop word removal
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


# Load and prepare dataset
excel_file_path = 'bug_dataset_w_dal.xlsx'

try:
    df = pd.read_excel(excel_file_path)
    print(f"Successfully loaded data from {excel_file_path}")
    print("Original columns:", df.columns.tolist())
    print("Original shape:", df.shape)
except FileNotFoundError:
    print(f"Error: File not found at {excel_file_path}")
    exit()
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()

# Select Columns and Combine Text
required_columns = ['Bug Title', 'Description', 'Labels']
if not all(col in df.columns for col in required_columns):
    print(f"Error: Missing one or more required columns: {required_columns}")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

# Combine title and description. Fill NaN with empty strings first.
df['text_input'] = df['Bug Title'].fillna('') + ' ' + df['Description'].fillna('')
print("\nCombined 'Bug Title' and 'Description' into 'text_input'.")

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Remove template phrases from the combined text
print("\nRemoving template phrases...")
df['text_input_no_template'] = df['text_input'].apply(remove_template_phrases)

# Apply the full preprocessing pipeline
print("Applying preprocessing (lowercase, decamelcase, stop words, lemmatization)...")
df['processed_text'] = df['text_input_no_template'].apply(
    lambda x: preprocess_text(x, stop_words, lemmatizer)
)
print("Preprocessing complete.")

# Ensure Labels column is string type before splitting, handle potential NaN
df['Labels'] = df['Labels'].fillna('')

# Split comma-separated labels into lists, removing leading/trailing whitespace
df['label_list'] = df['Labels'].apply(lambda x: [label.strip() for label in x.split(',') if label.strip()])
print("\nProcessed 'Labels' column into 'label_list'.")


# Select the columns for the final processed dataset
df_final = df[['processed_text', 'label_list']].copy()

print("\nSample of processed data:")
print(df_final.head())

# Save the processed data to a new file
output_file_path = 'bug_dataset_w_dal_preprocessed.csv'
df_final.to_csv(output_file_path, index=False)
print(f"\nProcessed data saved to {output_file_path}")