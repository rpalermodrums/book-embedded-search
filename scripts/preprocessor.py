import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_csv(input_path, output_path):
    """Preprocess the data by grouping chapters, removing stopwords and unimportant columns."""
    df = load_and_prepare_data(input_path)
    df = remove_stopwords(df)
    save_data(df, output_path)

def load_and_prepare_data(input_path):
    """Load the data and prepare it by grouping chapters and removing unimportant columns."""
    df = pd.read_csv(input_path)
    df['title'] = df['book_id'].apply(lambda x: x.split('.')[0])
    df_grouped = df.groupby(['title', 'book_id'])['summary'].apply(' '.join).reset_index()
    return df_grouped

def remove_stopwords(df):
    """Remove English stopwords from the summary column."""
    df['summary'] = df['summary'].apply(lambda x: ' '.join([word for word in x.split() if word not in ENGLISH_STOP_WORDS]))
    return df

def save_data(df, output_path):
    """Save the processed DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_csv('raw_data.csv', 'processed_data.csv')
