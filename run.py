from scripts import data_downloader, preprocessor, verbosity_reducer, embedding_generator, book_searcher
import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

if __name__ == "__main__":
    data_downloader.download_dataset("kmfoda/booksum", "raw_data.csv")
    preprocessor.preprocess_csv('raw_data.csv', 'processed_data.csv')
    verbosity_reducer.reduce_verbosity('processed_data.csv', 'minimal_processed_data.csv')
    embedding_generator.generate_embeddings('minimal_processed_data.csv', 'book_summaries_with_embeddings.csv')
    book_searcher.search_books('medieval sci-fi dystopia fiction')
