import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk

def reduce_verbosity(input_path, output_path):
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    lemmatizer = WordNetLemmatizer()

    df = pd.read_csv(input_path)
    df = process_summaries(df)
    df = extract_keywords(df)
    save_data(df, output_path)

def process_summaries(df):
    lemmatizer = WordNetLemmatizer()
    df['summary'] = df['summary'].apply(lambda x: ' '.join(process_summary(x, lemmatizer)))
    return df

def process_summary(summary, lemmatizer):
    tokens = word_tokenize(summary)
    lemmas = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    freq_dist = nltk.FreqDist(lemmas)
    return [word for word in freq_dist.keys() if freq_dist[word] > 1]

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def extract_keywords(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['summary'])
    feature_array = vectorizer.get_feature_names_out()  # Changed to get_feature_names_out

    df['summary'] = df['summary'].apply(lambda x: ' '.join(get_top_keywords(x, vectorizer, feature_array)))
    return df

def get_top_keywords(summary, vectorizer, feature_array):
    tfidf_sorting = np.argsort(vectorizer.transform([summary]).toarray()).flatten()[::-1]
    return [feature_array[i] for i in tfidf_sorting[:10]]

def save_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    reduce_verbosity('processed_data.csv', 'minimal_processed_data.csv')
