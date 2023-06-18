# LM Book Summaries Search

## Description
This project downloads a book summaries dataset, pre-processes it, generates embeddings for each book summary, and allows you to search the books based on a query.

## Scripts
* data_downloader.py: Downloads the book summaries dataset from Hugging Face and saves it as a CSV file.
* preprocessor.py: Pre-processes the data by grouping chapters, removing stop-words and unimportant columns.
* verbosity_reducer.py: Reduces verbosity of the summaries by lemmatizing, removing low frequency words and extracting top keywords.
* embedding_generator.py: Generates embeddings for each book summary using OpenAI's get_embedding function.
* book_searcher.py: Searches the books based on a query and prints the titles of the top 3 most similar books.
* run.py: Runs all scripts to download the dataset, pre-process it, generate embeddings, and search the books.

## How to Run
To run the entire project, use the command python run.py.

This will download the dataset, pre-process it, generate embeddings, and search the books based on the query 'medieval sci-fi dystopia fiction'. If you want to use a different query, you can change it in book_searcher.py.

## Dependencies
This project requires the following libraries: pandas, sklearn, nltk, tiktoken, openai, datasets, scipy. You can install them using pip:

```
pip install pandas scikit-learn nltk tiktoken openai datasets scipy
```

Please make sure that the necessary files ('raw_data.csv', 'processed_data.csv', 'minimal_processed_data.csv', 'book_summaries_with_embeddings.csv') are in the same directory as the scripts. If your file paths are different, you will need to update these paths in the scripts.

## TODO
* In preprocessing, collapse all rows by title up to the first ".", then append all words to make one row.
* Remove duplicate embeddings between modules, especially embedding_generator.py and book_searcher.py
* Make book_searcher.py as fast as possible as it's the user-facing element
* Make this generally smarter...
* QA
* Dockerize