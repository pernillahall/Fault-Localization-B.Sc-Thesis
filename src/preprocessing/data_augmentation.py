import nltk
import random
import pandas as pd
import ast
import numpy as np
import joblib
import itertools
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def get_label_freq(train_df):
    """Count label frequencies across training examples"""
    all_labels = list(itertools.chain.from_iterable(train_df['label_list']))
    return Counter(all_labels)

def get_low_rep_indices(train_df, threshold=25):
    """Return indices of samples including underrepresented labels"""
    label_freq = get_label_freq(train_df)
    low_rep_labels = {label for label, count in label_freq.items() if count < threshold}

    indices = train_df[train_df['label_list'].apply(lambda labels: any(label in low_rep_labels for label in labels))].index
    return indices

def get_synonyms(word):
    """Function to retreive a word synonym from WordNet"""
    synonyms = set()
    for s in wordnet.synsets(word):
        for l in s.lemmas():
            synonym = l.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(text, n=1):
    """Function to perform synonym replacement on an input text"""
    words = text.split()
    if len(words) < 1:
        return text
    words_to_replace = random.sample(words, min(n, len(words)))
    new_words = words.copy()
    for word in words_to_replace:
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if w == word else w for w in new_words]
    return ' '.join(new_words)

def random_swap(text, n=1):
    """Function to perform random swap on an input text"""
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def full_augment_train_df(train_df, method='sr', factor=1):
    """Function to augment a dataset"""
    aug_rows = []

    for _, row in train_df.iterrows():
        for _ in range(factor):
            if method == 'sr':
                augmented_text = synonym_replacement(row['processed_text'], n=2)
            elif method == 'rs':
                augmented_text = random_swap(row['processed_text'], n=2)
            else:
                raise ValueError('Unknown augmentation method.')
            
            aug_rows.append({
                'processed_text': augmented_text,
                'label_list': row['label_list']
            })

    augmented_df = pd.DataFrame(aug_rows)
    return pd.concat([train_df, augmented_df], ignore_index=True)

def augment_underrepresented(train_df, method='sr', factor=1, threshold=20):
    """Augment only samples with underrepresented labels."""
    underrep_indices = get_low_rep_indices(train_df, threshold=threshold)
    aug_rows = []

    for idx in underrep_indices:
        row = train_df.loc[idx]
        for _ in range(factor):
            if method == 'sr':
                augmented_text = synonym_replacement(row['processed_text'], n=2)
            elif method == 'rs':
                augmented_text = random_swap(row['processed_text'], n=2)
            else:
                raise ValueError('Unknown augmentation method.')
            
            aug_rows.append({
                'processed_text': augmented_text,
                'label_list': row['label_list']
            })

    augmented_df = pd.DataFrame(aug_rows)
    return pd.concat([train_df, augmented_df], ignore_index=True)


def encode_and_save(train_df, val_df, test_df, prefix):
    """Function to encode dataset labels and save to file"""
    label_set = sorted({label for labels in train_df['label_list'] for label in labels})
    mlb = MultiLabelBinarizer(classes=label_set)
    mlb.fit(train_df['label_list'])

    y_train = mlb.transform(train_df['label_list'])
    y_val = mlb.transform(val_df['label_list'])
    y_test = mlb.transform(test_df['label_list'])

    np.save(f"data/prepared_data/y_train_bin{prefix}.npy", y_train)
    np.save(f"data/prepared_data/y_val_bin{prefix}.npy", y_val)
    np.save(f"data/prepared_data/y_test_bin{prefix}.npy", y_test)
    joblib.dump(mlb, f"data/prepared_data/mlb{prefix}.joblib")


# Load original splits
train_df = pd.read_csv("data/prepared_data/train_df.csv")
val_df = pd.read_csv("data/prepared_data/val_df.csv")
test_df = pd.read_csv("data/prepared_data/test_df.csv")

# Convert label strings to list
for df in [train_df, val_df, test_df]:
    if isinstance(df['label_list'].iloc[0], str):
        df['label_list'] = df['label_list'].apply(ast.literal_eval)

# Perform augmentation and save sets
train_df_sr = full_augment_train_df(train_df, method='sr')
train_df_rs = full_augment_train_df(train_df, method='rs')

train_df_sr_underrep = augment_underrepresented(train_df, method='sr', factor=2, threshold=25)
train_df_rs_underrep = augment_underrepresented(train_df, method='rs', factor=2, threshold=25)

train_df_sr.to_csv("data/prepared_data/train_df_SR.csv", index=False)
train_df_rs.to_csv("data/prepared_data/train_df_RS.csv", index=False)

train_df_sr_underrep.to_csv("data/prepared_data/train_df_SR_underrep.csv", index=False)
train_df_rs_underrep.to_csv("data/prepared_data/train_df_RS_underrep.csv", index=False)

# Save for each dataset
encode_and_save(train_df_sr, val_df, test_df, "_SR")
encode_and_save(train_df_rs, val_df, test_df, "_RS")

encode_and_save(train_df_sr_underrep, val_df, test_df, "_SR_underrep")
encode_and_save(train_df_rs_underrep, val_df, test_df, "_RS_underrep")