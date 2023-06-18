import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import get_embedding

def generate_embeddings(input_path, output_path):
    """Generate embeddings for each book summary."""
    df = load_and_prepare_data(input_path)
    df = calculate_embeddings(df)
    save_data(df, output_path)

def load_and_prepare_data(input_path):
    """Load the data and prepare it by omitting summaries that are too long to embed."""
    df = pd.read_csv(input_path)
    df = df.dropna()
    df["combined"] = "Title: " + df["book_id"].str.strip() + "; Content: " + df["summary"].str.strip()
    max_tokens = 8000
    encoding = tiktoken.get_encoding("cl100k_base")
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    return df[df.n_tokens <= max_tokens]

def calculate_embeddings(df):
    """Calculate embeddings for each book summary."""
    df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine="text-embedding-ada-002"))
    return df

def save_data(df, output_path):
    """Save the DataFrame with embeddings to a CSV file."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    generate_embeddings('minimal_processed_data.csv', 'book_summaries_with_embeddings.csv')
