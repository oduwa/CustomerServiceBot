import re,sys
import nltk.tokenize
import unittest

class InfoExtractor(object):

    def __pairwise_word_distance_in_text(self, word1, word2, text):
        all_words = nltk.tokenize.word_tokenize(text)
        print all_words
        idx1 = all_words.index(word1)+1 # +1 to account for zero indexing
        idx2 = all_words.index(word2)+1 # +1 to account for zero indexing
        return abs(idx1-idx2)

    def __extract_account_no(self, text):
        # Lowercase
        lower_text = text.lower()

        # Find a match for an 8 digit number
        matches = re.findall(r'\b([0-9]{8})\b', lower_text)

        # Assume only one account no. is to be returned as per the task specification
        # As such, when more than one 8 digit number is found, select 8 digit number closest to relevant words.
        # Even if two instances of the word "account" exist, the account no closest to first occurence
        # of account will definitely be a valid account number. However, multiple account numbers will not be captured
        # (as that is out of the scope of this task).
        closestMatch = ""
        closestDistance = sys.maxint
        for m in matches:
            distance = self.__pairwise_word_distance_in_text(m, "account", lower_text)
            if(distance < closestDistance):
                closestDistance = distance
                closestMatch = m
        if not closestMatch:
            return None
        else:
            return closestMatch

    def __extract_order_id(self, text):
        # Find a match for an 8 digit alphanumeric string
        matches = re.findall(r'(?![0-9]{8})\b[a-zA-Z\d]{8}\b', text)

        # Assume only one order id is to be returned as per the task specification.
        # As such, when more than one 8 digit alphanumeric (containing both numbers AND letters)
        # is found, select 8 digit alphanumeric closest to relevant words.
        # Even if two instances of the word "order" appear, the id closest to first occurence
        # of "order" will definitely be a valid order number. However, multiple order numbers will not be captured
        # (as that is out of the scope of this task).
        closestMatch = ""
        closestDistance = sys.maxint
        for m in matches:
            distance = self.__pairwise_word_distance_in_text(m.lower(), "id", text.lower())
            if(distance < closestDistance):
                closestDistance = distance
                closestMatch = m
        if not closestMatch:
            return None
        else:
            return closestMatch

    def extract_info(self, text):
        '''
        Extracts account number and order id from a given message text.

        @param text message text to extract order id and account number from.

        @return Dict dictionary of the form

            {
                "order_id":<XXXXXXXX>,
                "account_number":<XXYYXXYY>
            }
        '''
        result = {}
        result["order_id"] = self.__extract_order_id(text)
        result["account_number"] =self.__extract_account_no(text)
        return result

class InfoExtractorTestCase(unittest.TestCase):
    def setUp(self):
        self.extractor = InfoExtractor()

    def test_extract_info(self):
        self.assertEqual(self.extractor.extract_info('account number is 72934923,\
         and order id is 987BGF12'), {"order_id":"987BGF12","account_number":"72934923"})
        self.assertEqual(self.extractor.extract_info('account number 01928340'), {"order_id":None,"account_number":"01928340"})
        self.assertEqual(self.extractor.extract_info('no account number or id'), {"order_id":None,"account_number":None})
        self.assertEqual(self.extractor.extract_info('order id 372872873287'), {"order_id":None,"account_number":None})


if __name__ == '__main__':
    unittest.main()
