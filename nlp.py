from langdetect import detect, DetectorFactory
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
import nltk
import nltk
from nltk.corpus import wordnet as wn
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# text = input("Enter text: ")

DetectorFactory.seed = 0
print(detect("Ana are mere"))

def stilometric_info(text):
    # Tokenizare text
    words = word_tokenize(text)
    
    # Lungimea în cuvinte și caractere
    num_words = len(words)
    num_chars = len(text)
    
    # Frecvența cuvintelor
    freq_dist = FreqDist(words)
    
    # Afișare informații
    print(f"Lungimea în cuvinte: {num_words}")
    print(f"Lungimea în caractere: {num_chars}")
    print("Frecvența cuvintelor:")
    for word, freq in freq_dist.items():
        print(f"{word}: {freq}")

text = "Acesta este un exemplu de text, pentru a afișa informații stilometrice folosind nltk."

# stilometric_info(text)
from nltk.tokenize import word_tokenize

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def get_hypernyms(word):
    hypernyms = set()
    for syn in wn.synsets(word):
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                hypernyms.add(lemma.name())
    return list(hypernyms)

def get_antonym_negation(word):
    antonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.add("nu " + lemma.antonyms()[0].name())
    return list(antonyms)

def generate_alternative_versions(text, replacement_rate=0.2):
    words = word_tokenize(text)
    new_texts = []
    for _ in range(5):  # Generăm 5 versiuni alternative
        new_words = []
        for word in words:
            if random.random() < replacement_rate:
                synonyms = get_synonyms(word)
                hypernyms = get_hypernyms(word)
                antonyms = get_antonym_negation(word)
                replacements = synonyms + hypernyms + antonyms
                if replacements:
                    new_words.append(random.choice(replacements))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        new_texts.append(" ".join(new_words))
    return new_texts

# Exemplu de text
text = "Hi. I am a student. I like to study natural language processing. It is interesting. I am learning new things."

# Generare versiuni alternative
alternative_versions = generate_alternative_versions(text)

# Afișare versiuni alternative
for i, version in enumerate(alternative_versions, 1):
    print(f"Versiunea {i}: {version}")

def extract_keywords(text, num_keywords=5):
    stop_words = set(stopwords.words('romanian'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_words)])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    keyword_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
    keywords = [feature_names[i] for i in keyword_indices]
    
    return keywords

def generate_sentences(text, keywords):
    sentences = nltk.sent_tokenize(text)
    keyword_sentences = {}
    
    for keyword in keywords:
        for sentence in sentences:
            if keyword in sentence.lower():
                keyword_sentences[keyword] = sentence
                break
    
    return keyword_sentences

# Exemplu de text
text = "Acesta este un exemplu de text pentru a afișa informații stilometrice folosind nltk. NLTK este o bibliotecă puternică pentru procesarea limbajului natural."

# Extrage cuvintele cheie
keywords = extract_keywords(text)
print("Cuvinte cheie:", keywords)

# Generează propoziții pentru cuvintele cheie
keyword_sentences = generate_sentences(text, keywords)
for keyword, sentence in keyword_sentences.items():
    print(f"Propoziție pentru '{keyword}': {sentence}")

