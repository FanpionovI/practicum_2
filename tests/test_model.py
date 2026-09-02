import unittest

from spam_model import detect_language, predict_messages, risk_signals, train_model


class SpamModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = train_model()

    def test_supported_languages_are_detected(self):
        self.assertEqual(detect_language("Вы получили подарок"), "ru")
        self.assertEqual(detect_language("¡Has ganado un premio!"), "es")
        self.assertEqual(detect_language("See you tonight"), "en")

    def test_clear_multilingual_spam_examples(self):
        messages = [
            "СРОЧНО! Вы выиграли 50000 рублей, перейдите по ссылке prize.example!",
            "WINNER! Claim your free £1000 cash prize at reward.example now!",
            "¡URGENTE! Has ganado 1000 euros. Reclama tu premio gratis ahora!",
        ]
        self.assertTrue(predict_messages(self.bundle, messages).prediction.eq("spam").all())

    def test_clear_multilingual_ham_examples(self):
        messages = [
            "Привет, я немного задержусь, встречаемся у метро в семь.",
            "I will be ten minutes late, see you near the station.",
            "Ana, nos vemos en la oficina mañana por la tarde.",
        ]
        self.assertTrue(predict_messages(self.bundle, messages).prediction.eq("ham").all())

    def test_risk_signals(self):
        signals = risk_signals("СРОЧНО! Вы выиграли 50 000 ₽! prize.example")
        self.assertTrue({"link", "money", "urgency", "reward", "exclamations"}.issubset(signals))


if __name__ == "__main__":
    unittest.main()

