import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import wordnet as wn
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from colorama import init, Fore
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

init(autoreset=True)

class NLPProcessor:
    def __init__(self, text):
        self.text = text
        self.words = word_tokenize(text)
        self.num_words = len(self.words)
        self.num_chars = len(text)
        self.freq_dist = FreqDist(self.words)

    def stilometric_info(self):
        print(Fore.BLUE + "Lungimea în cuvinte:" + f"{self.num_words}")
        print(Fore.BLUE + "Lungimea în caractere:"+ f"{self.num_chars}")
        print(Fore.BLUE + "Frecvența cuvintelor:")
        for word, freq in self.freq_dist.items():
            print(f"{word}: {freq}")
        print()

    @staticmethod
    def get_synonyms(word):
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    @staticmethod
    def get_hypernyms(word):
        hypernyms = set()
        for syn in wn.synsets(word):
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    hypernyms.add(lemma.name())
        return list(hypernyms)

    def generate_alternative_versions(self, max_percentage=0.2):
        alternative_versions = []
        num_replacements = int(self.num_words * max_percentage)
        words_to_replace = random.sample(self.words, num_replacements)
        
        new_words = self.words.copy()
        for i, word in enumerate(self.words):
            if word in words_to_replace:
                synonyms = self.get_synonyms(word)
                hypernyms = self.get_hypernyms(word)
                replacements = synonyms + hypernyms
                if replacements:
                    new_words[i] = random.choice(replacements)
        
        new_text = ' '.join(new_words)
        return new_text

    def extract_keywords(self, num_keywords=5):
        stop_words = set(stopwords.words())
        filtered_words = [word for word in self.words if word.isalnum() and word not in stop_words]
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_words)])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        keyword_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
        keywords = [feature_names[i] for i in keyword_indices]
        
        return keywords

    def generate_sentences(self, keywords):
        sentences = nltk.sent_tokenize(self.text)
        keyword_sentences = {}
        
        for keyword in keywords:
            for sentence in sentences:
                if keyword in sentence.lower():
                    keyword_sentences[keyword] = sentence
                    break
        
        return keyword_sentences

# Example usage
if __name__ == "__main__":
    text = """
        My cat, Whiskers, loves to chase yarn. I think it's because she's part bobcat.
        Whiskers has always been a bit wild, with her sharp instincts and playful nature. 
        She often stalks her toys like a true predator, and her favorite game is hide and seek. 
        She hides behind furniture and waits for the perfect moment to pounce.
        It's always a surprise when she leaps out with a playful attack.
    """
    print(Fore.BLUE + "Text:", text)

    processor = NLPProcessor(text)
    # processor.stilometric_info()
    print(Fore.BLUE + "Synonyms for 'yarn':", processor.get_synonyms('yarn'))

    # Generare versiuni alternative
    alternative_versions = processor.generate_alternative_versions()
    print(Fore.BLUE + "Alternative versions:", alternative_versions)

    keywords = processor.extract_keywords()
    print(Fore.BLUE + "Keywords:")
    print(keywords)

    keyword_sentences = processor.generate_sentences(keywords)
    for keyword, sentence in keyword_sentences.items():
        print(Fore.BLUE + f"Sentence for '{keyword}': ")
        print(sentence.strip())
