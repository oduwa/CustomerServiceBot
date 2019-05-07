from nltk.corpus import stopwords
import nltk.tokenize
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import csv
import re,sys
import Helper
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import argparse
import random
from sklearn.metrics.pairwise import cosine_similarity
import unittest


TRAIN_RATIO = 1.0
MODEL_PATH = "model/intent_mlp.ser"
VOCAB_PATH = "model/vocabulary.ser"
DATA_PATH = "data/dataset.ser"
class_mapping = {'cancel_order':0, 'order_status':1, 'unknown':2}
reverse_class_mapping = {0:'cancel_order', 1:'order_status', 2:'something_else'}
source_mapping = {'agent':0, 'customer':1}

class IntentDetector(object):
    def __load_data(self):
        '''
        Load dataset from CSV file.

        Loads customer message data from the customer message data provided at
        data/tech_test_data.csv. Loads another small dataset of Twitter messages to
        augments this dataset with examples for "something else" messages.

        @return List<Dict> List of message objects modelled as dictionary objects with
                            the keys "text", "source" and "type".
        '''
        messages = []

        # LOAD DATA FROM DATASET
        with open('data/tech_test_data.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                msg = {}
                msg['text'] = Helper.preprocess_text(row['message'])
                msg['source'] = row['message_source']
                msg['type'] = row['case_type']
                msg['conversation_id'] = row['conversation_id']
                msg['message_number'] = row['message_number']
                messages.append(msg)

        # AUGMENT DATA WITH "SOMETHING ELSE" INTENT EXAMPLES USING GENERIC SOCIAL MEDIA MESSAGES
        with open('data/social_media_data.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            i = 0
            number_unknown = 100
            for row in reader:
                msg = {}
                msg['text'] = Helper.preprocess_text(row['text'])
                msg['source'] = 'customer'
                msg['type'] = 'unknown'
                msg['conservation_id'] = None
                msg['message_number'] = None
                messages.append(msg)
                i += 1
                if i == number_unknown:
                    break

        # SHUFFLE
        random.seed(42)
        random.shuffle(messages)
        return messages

    def __build_vocabulary(self, dataset):
        '''
        Builds vocabulary of all words that appear in the corpus

        @param List<Dict> dataset List of message objects modelled as dictionary objects with
                            the keys "text", "source" and "type"

        @return List<String> A list with unique entries, containing
                            all words that appear in the dataset
        '''
        # CREATE TERM DICTIONARY
        all_tokens = []
        for msg in dataset:
            ## TODO: TRY REMOVING IDS FROM TERM DICTIONARY AS THEY DO NOT AID IN CLASSIFICATION
            tokenized_msg = nltk.tokenize.word_tokenize(msg['text'])
            all_tokens.extend(tokenized_msg)
        term_dictionary = list(set(all_tokens))

        return term_dictionary

    def __build_vocabulary_bigram(self, dataset):
        '''
        Builds vocabulary of all bigrams that appear in the corpus

        @param List<Dict> dataset List of message objects modelled as dictionary objects with
                            the keys "text", "source" and "type"

        @return List<String> A list with unique entries, containing
                            all bigrams that appear in the dataset
        '''
        # CREATE TERM DICTIONARY
        all_bigrams = []
        for msg in dataset:
            ## TODO: TRY REMOVING IDS FROM TERM DICTIONARY AS THEY DO NOT AID IN CLASSIFICATION
            tokenized_msg = nltk.tokenize.word_tokenize(msg['text'])
            for i in range(len(tokenized_msg)-1):
                bigram = tokenized_msg[i] + " " + tokenized_msg[i+1]
                all_bigrams.append(bigram)
        term_dictionary = list(set(all_bigrams))

        return term_dictionary

    def __bag_of_word_features(self, msg, term_dictionary):
        '''
        Constructs a 1-gram feature representation for a given message, using the
        corpus vocabulary.

        @param Dict msg dictionary object representing the message to create vector for
                            with the keys "text", "source" and "type".
        @param List<String> corpus vocabulary. A list with unique entries, containing
                            all words that appear in the dataset.

        @return Dict n-gram feature vector. A dictionary where each key is a word in
                the vocabulary and its value is the number of times it appears in the text.
        '''
        text_tokens = nltk.tokenize.word_tokenize(msg['text'])
        features = {}

        # Initialize occurence occurence counts to zero
        for token in term_dictionary:
            features[token] = 0

        # Count word occurence frequencies
        for token in term_dictionary:
            if(token in text_tokens):
                features[token] = features[token] + 1
        features['message_source'] = source_mapping[msg['source']]

        return features

    def __bigram_features(self, msg, term_dictionary):
        '''
        Constructs a bigram feature representation for a given message, using the
        corpus vocabulary.

        @param Dict msg dictionary object representing the message to create vector for
                            with the keys "text", "source" and "type".
        @param List<String> corpus vocabulary. A list with unique entries, containing
                            all words that appear in the dataset.

        @return Dict bigram feature vector. A dictionary where each key is a bigram in
                the vocabulary and its value is the number of times it appears in the text.
        '''
        text_tokens = Helper.bigram_tokenize(msg['text'])
        features = {}

        # Initialize occurence occurence counts to zero
        for token in term_dictionary:
            features[token] = 0

        # Count word occurence frequencies
        for token in term_dictionary:
            if(token in text_tokens):
                features[token] = features[token] + 1
        features['message_source'] = source_mapping[msg['source']]

        return features

    def __build_model(self, dataset):
        '''
        Train model (or load from disk if already previously trained).

        @param List<Dict> dataset List of message objects modelled as dictionary objects with
                            the keys "text", "source" and "type"

        @return (classifier, vocabulary)
        '''
        # BUILD VOCAB
        vocabulary = None
        if not Helper.is_serialized_object_in_path(VOCAB_PATH):
            vocabulary = self.__build_vocabulary(dataset)
        else:
            vocabulary = Helper.unserialize(VOCAB_PATH)

        if not Helper.is_serialized_object_in_path(MODEL_PATH):
            # CONSTRUCT FEATURE SETS
            featureset = [(self.__bag_of_word_features(msg, vocabulary), class_mapping[msg['type']]) for msg in dataset]
            train_set = featureset[:int(len(featureset)*TRAIN_RATIO)]
            test_set = featureset[int(len(featureset)*TRAIN_RATIO):]

            # TRAIN CLASSIFIER AND EVALUATE
            classifier = SklearnClassifier(MLPClassifier()).train(train_set)
        else:
            classifier = Helper.unserialize(MODEL_PATH)

        return classifier, vocabulary

    def __get_message_response_pairs(self):
        '''
        Collects and returns pairs of customer messages and the corresponding agent
        responses.

        @return List<(MESSAGE, RESPONSE)> pairs Returns a list of tuples where the
        first element in the tuple is a message dictionary object and the second
        is a response dictionary object.
        '''
        # LOAD DATASET
        if Helper.is_serialized_object_in_path(DATA_PATH):
            dataset = Helper.unserialize(DATA_PATH)
        else:
            dataset = self.__load_data()

        msg_response_table = {}
        msg_response_pairs = []
        for i in range(len(dataset)):
            if dataset[i]['message_number'] == '1':
                j = i+1
                while j < len(dataset):
                    if dataset[j]['source'] == 'agent':
                        msg_response_pairs.append((dataset[i], dataset[j]))
                    j+=1
        return msg_response_pairs

    def suggest_reply(self,message):
        '''
        Suggests a reply to a given message.
        Does this by finding the most similar message to the input message
        within the training dataset, and returning the reply for that message.

        @param String message A message to get a reply for.
        @return String reply for input message.
        '''
        # Get message-response pairs
        msg_response_pairs = self.__get_message_response_pairs()

        # Get customer texts together with indices back to msg_response_pairs
        indexed_texts = [(msg_pair[0],idx) for idx,msg_pair in enumerate(msg_response_pairs) if msg_pair[0]['source'] == 'customer']

        # VECTORIZE
        vectorizer = CountVectorizer()
        vecs = vectorizer.fit_transform([msg['text'] for msg,idx in indexed_texts] + [message])

        # Compute most similar
        test_vec = vecs[-1]
        vecs = vecs[:vecs.shape[0]-1]
        closestIdx = 0
        biggestSimilarity = sys.float_info.epsilon
        for i in range(vecs.shape[0]):
            v = vecs[i]
            sim = cosine_similarity(v,test_vec)[0][0]
            if sim > biggestSimilarity:
                biggestSimilarity = sim
                closestIdx = i

        # Return reply for most similar train message to input message
        closest_pair = msg_response_pairs[indexed_texts[closestIdx][1]]
        return closest_pair[1]['text']



    def classify_message_intent(self, message):
        '''
        Predict the intent of a given message object

        @param Dict message dictionary objects with the keys "text" and "source".

        @return Dict result dictionary where each key is a possible class and the values are
                    corresponding class probability predictions.
        '''
        # LOAD DATASET
        if Helper.is_serialized_object_in_path(DATA_PATH):
            dataset = Helper.unserialize(DATA_PATH)
        else:
            dataset = self.__load_data()

        # BUILD MODEL
        classifier, vocabulary = self.__build_model(dataset)

        # CLASSIFY WITH MODEL
        message_feature_vector = self.__bag_of_word_features(message, vocabulary)
        class_probabilities = classifier.prob_classify(message_feature_vector)

        prediciton_idx = class_probabilities.max()
        prediction_label = reverse_class_mapping[prediciton_idx]

        result = {"prediction":prediction_label, "confidence":class_probabilities.prob(prediciton_idx)}
        if(prediciton_idx != 2):
            result['suggested_message'] = self.suggest_reply(message['text'])
        return result

class IntentDetectorTestCase(unittest.TestCase):
    def setUp(self):
        self.detector = IntentDetector()

    def test_classify_message_intent(self):
        self.assertEqual(self.detector.classify_message_intent({"text":"Can i \
        cancel my order please","source":"customer"}), {"prediction":"cancel_order"\
        , "confidence":0.7452696875281501, "suggested_message":"of course! let me assist. please share your account number and order id and i'll see what the options are."})
        self.assertEqual(self.detector.classify_message_intent({"text":"my name is oduwa",\
        "source":"customer"}), {"confidence": 0.5170547250874366, "prediction": "something_else"})

    def test_suggest_reply(self):
        self.assertEqual(self.detector.suggest_reply("can i cancel my order please"), "of course! let me assist. please share your account number and order id and i'll see what the options are.")
        self.assertEqual(self.detector.suggest_reply("when will my order arrive"), "of course! let me assist. please share your account number and order id and i'll see what the options are.")
        self.assertEqual(self.detector.suggest_reply("my order number is XXXXXXXX"), "thank you! please wait a minute while i check your order status.")
        self.assertEqual(self.detector.suggest_reply("account number X5X5X5X5 order number is XXXXXXXX"), "thank you! please wait a minute while i check your order status.")

if __name__ == "__main__":
    unittest.main()
