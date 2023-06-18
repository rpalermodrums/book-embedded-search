import data_downloader
import preprocessor
import verbosity_reducer
import embedding_generator
import book_searcher

def run():
    """Run all scripts to download the dataset, preprocess it, generate embeddings, and search the books."""
    data_downloader.download_dataset("kmfoda/booksum", "raw_data.csv")
    preprocessor.preprocess_csv('raw_data.csv', 'processed_data.csv')
    verbosity_reducer.reduce_verbosity('processed_data.csv', 'minimal_processed_data.csv')
    embedding_generator.generate_embeddings('minimal_processed_data.csv', 'book_summaries_with_embeddings.csv')
    book_searcher.search_books('medieval sci-fi dystopia fiction')

if __name__ == "__main__":
    run()
