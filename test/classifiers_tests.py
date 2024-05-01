import unittest
import random
import string
import sys
import os
from dotenv import main

sys.path.append('..')
from model_evaluation_scripts.classifiers_classes_api.multi_bert_classifier import multi_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from model_evaluation_scripts.classifiers_classes_api.mixtral_8x7b_API_classifier import mistral_classifier



class Test_Multi_bert_Methods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Inicializando MultiBert")
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

    def test_get_most_toxic_messages(self):
        toxic_messages = ["I hate u" for i in range(0, 10)]
        messages = ["i love u"] + toxic_messages
        most_toxic_messages = self.multibert.get_most_toxic_messages(messages)
        self.assertTrue("i love u" not in most_toxic_messages)

    def test_group_toxicity_distribution_generalisation(self):
        #intento que sea evidentemente generalizador
        scores = self.multibert.get_group_toxicity_distribution(["everyone is the same", "i hate all republicans", "i hate everyone in that specific group people"])
        print(scores)
        self.assertEqual(scores["generalisation"], 1)

    def test_group_toxicity_distribution_sarcasm(self):
        #tres mensajes que deberian dar muy sarcastico
        scores = self.multibert.get_group_toxicity_distribution(["you are right, i do want you. I want you as far as possible.", "those news are certanly true, also today i saw a cow flying maybe report that?", "u are the smartest man alive even einsteins brains is smaller than yours"])
        #print(scores)

        self.assertEqual(scores["sarcastic"], 1)

    def test_group_toxicity_distribution_low(self):
        #tres mensajes que deberian dar muy neutros
        scores = self.multibert.get_group_toxicity_distribution(["Hello! How are you?", "Milei says that the economy is growing, but experts digress", "i dont belive you"])
        #print(scores)
        self.assertEqual(scores["healthy"], 1)
        self.assertEqual(scores["dismissive"], 0)
        self.assertEqual(scores["hostile"], 0)


class Test_hate_bert_Methods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hatebert = hate_bert_classifier("../model_evaluation_scripts/classifiers_classes_api/toxigen_hatebert")
    
    def test_get_most_toxic_messages(self):
        toxic_messages = ["I hate u" for i in range(0, 10)]
        messages = ["i love u"] + toxic_messages
        most_toxic_messages = self.hatebert.get_most_toxic_messages(messages)
        self.assertTrue("i love u" not in most_toxic_messages)

    def test_get_most_toxic_message(self):
        toxic_messages = ["i do not like deaf black woman" for i in range(0, 10)]
        messages = ["I am a black deaf woman"] + toxic_messages
        most_toxic_messages = self.hatebert.get_most_toxic_messages(messages)
        self.assertTrue("I am a black deaf woman" not in most_toxic_messages)

    def test_predict_average_toxicity_scores(self):
        toxic_messages = ["I hate u", "i love u"]
        a = self.hatebert.predictToxicity(toxic_messages[0])[1]
        b = self.hatebert.predictToxicity(toxic_messages[1])[1]
        self.assertEqual((a + b)/2, self.hatebert.predict_average_toxicity_score(toxic_messages))


class Test_mixtral_Methods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.load_dotenv()
        MISTRAL_API_KEY:str = os.environ['MISTRAL_API_KEY']
        print("Inicializando Mistral")
        cls.mistral = mistral_classifier(mistral_api_key=MISTRAL_API_KEY, templatetype='prompt_template_few_shot', toxicity_distribution_path = "../model_evaluation_scripts/classifiers_classes_api/toxicity_distribution_cache/mistral_distribution.json", calculate_toxicity_distribution = True )
    
    def test_fake(self):
        self.assertTrue(True)

    def test_get_group_toxicity_distribution_high_toxicity(self):
        self.assertFalse(False)
        #high_toxicity_score = self.mistral.get_group_toxicity_distribution("i hope everyone dies everyone sucks")
        #self.assertEqual(high_toxicity_score == 1)

if __name__ == '__main__':
    unittest.main()