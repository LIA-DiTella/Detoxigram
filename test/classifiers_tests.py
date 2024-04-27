import unittest
import random
import string
import sys
sys.path.append('..')
from model_evaluation_scripts.classifiers_classes_api.multi_bert_classifier import multi_bert_classifier


class Test_Multi_bert_Methods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.multibert = multi_bert_classifier("../model_evaluation_scripts/classifiers_classes_api/multibert", toxicity_distribution_path = "../model_evaluation_scripts/classifiers_classes_api/toxicity_distribution_cache/multibert_distribution.json", calculate_toxicity_distribution = False)
    
    def test_unhealthy_items_range(self):
        for i in range(0, 30):
            random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k= 20))
            unhealthy_items = self.multibert.predict_toxicity_scores(random_string)[1]
            for l, v in unhealthy_items.items():
                self.assertGreater(v, 0)
                self.assertLessEqual(v, 1)

    def test_toxicity_range(self):
        for i in range(0, 30):
            random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k= 20))
            toxicity = self.multibert.predictToxicity(random_string)[1]
            self.assertGreater(toxicity, 0)
            self.assertLessEqual(toxicity, 4)

    def test_basic_none_toxicity(self):
        toxicity = self.multibert.predictToxicity("I like cats")[0]
        self.assertFalse(toxicity)

    def test_get_most_toxic_messages_test(self):
        toxic_messages = ["I hate u" for i in range(0, 10)]
        messages = toxic_messages + ["i love u"]
        most_toxic_messages = self.multibert.get_most_toxic_messages(messages)
        self.assertTrue("i love u" not in most_toxic_messages)

if __name__ == '__main__':
    unittest.main()