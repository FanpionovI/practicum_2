"""Training and inference for the multilingual SpamGuard prototype."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection"
RU_DATA_PATH = ROOT / "data" / "anti_spam_ru.csv"
LANGUAGE_NAMES = {"en": "English", "ru": "Русский", "es": "Español"}


@dataclass(frozen=True)
class ModelBundle:
    model: Pipeline
    metrics: dict
    dataset: pd.DataFrame
    test_results: pd.DataFrame


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def _render_templates(templates: list[str], values: dict[str, list[str]]) -> list[str]:
    rows: list[str] = []
    width = max(len(items) for items in values.values())
    for template_index, template in enumerate(templates):
        for value_index in range(width):
            payload = {key: items[(value_index + template_index) % len(items)] for key, items in values.items()}
            rows.append(template.format(**payload))
    return rows


def build_synthetic_multilingual_data() -> pd.DataFrame:
    """Small, disclosed RU/ES extension used only for the educational prototype."""
    ru_values = {
        "name": ["Анна", "Миша", "Олег", "Катя", "Ирина", "Дима", "Лена", "Саша"],
        "time": ["18:00", "семь вечера", "после работы", "завтра утром", "в 14:30", "через час"],
        "place": ["у метро", "в офисе", "дома", "в кафе", "у входа", "в университете"],
        "amount": ["50 000", "100 000", "25 000", "300 000", "75 000", "10 000"],
        "code": ["WIN", "ПРИЗ", "ДА", "BONUS", "ПОДАРОК", "CASH"],
        "phone": ["8-800-555-12-34", "9000", "8-900-123-45-67", "7777", "8-495-000-11-22", "9090"],
    }
    ru_ham = _render_templates(
        [
            "{name}, встречаемся {place} {time}?",
            "Я задерживаюсь, буду {time}. Позвони, когда приедешь.",
            "Заказ готов и ожидает вас {place}. Спасибо за покупку.",
            "Привет! Пришли, пожалуйста, конспект после пары.",
            "Напоминаем о записи к врачу {time}. Для переноса позвоните в регистратуру.",
            "Оплата получена. Электронный чек доступен в приложении банка.",
            "{name}, я уже {place}. Ты скоро?",
            "Занятие перенесли на {time}, аудитория осталась прежней.",
        ],
        ru_values,
    )
    ru_spam = _render_templates(
        [
            "СРОЧНО! Вы выиграли {amount} рублей. Отправьте {code} на номер {phone} прямо сейчас!",
            "Поздравляем, ваш номер выбран победителем! Заберите бесплатный приз: http://gift-now.example/{code}",
            "Ваша карта заблокирована. Немедленно подтвердите данные по ссылке bank-check.example/{code}",
            "Только сегодня займ без отказа до {amount} рублей! Оформить: money-fast.example",
            "Получите подарок БЕСПЛАТНО. Для активации отправьте код {code} на {phone}.",
            "Вам начислена компенсация {amount} руб. Оплатите комиссию для получения выплаты.",
            "Удалённая работа без опыта, доход {amount} в неделю. Пишите в Telegram прямо сейчас!",
            "Последнее предупреждение! Ваш аккаунт будет удалён. Войдите по ссылке verify-user.example.",
            "Эксклюзивная акция: скидка 90% только 10 минут. Нажмите ссылку и заберите товар!",
            "Вы получили бонус. Сообщите код из SMS оператору по номеру {phone}.",
        ],
        ru_values,
    )

    es_values = {
        "name": ["Ana", "Luis", "Marta", "Carlos", "Elena", "Diego", "Sofía", "Pablo"],
        "time": ["a las 18:00", "mañana", "esta tarde", "en una hora", "a las 14:30", "el lunes"],
        "place": ["en la estación", "en casa", "en la oficina", "en el café", "en la universidad", "en la entrada"],
        "amount": ["500 €", "1.000 €", "250 €", "5.000 €", "750 €", "10.000 €"],
        "code": ["PREMIO", "SI", "BONO", "GANA", "REGALO", "CASH"],
        "phone": ["900 123 456", "5050", "800 555 777", "7070", "910 000 111", "9090"],
    }
    es_ham = _render_templates(
        [
            "{name}, ¿nos vemos {place} {time}?",
            "Voy con retraso. Te llamo cuando llegue {place}.",
            "Tu pedido está listo para recoger {place}. Gracias por tu compra.",
            "Hola, ¿puedes enviarme los apuntes de la clase?",
            "Recordatorio de tu cita médica {time}. Llama para cambiarla.",
            "Pago recibido. El recibo está disponible en la aplicación.",
            "{name}, ya estoy {place}. ¿Llegas pronto?",
            "La reunión se ha movido {time}; la sala no cambia.",
        ],
        es_values,
    )
    es_spam = _render_templates(
        [
            "¡URGENTE! Has ganado {amount}. Envía {code} al {phone} ahora mismo.",
            "¡Felicidades! Tu número ha ganado un premio gratis. Reclámalo: http://premio.example/{code}",
            "Tu cuenta está bloqueada. Confirma tus datos inmediatamente en banco-seguro.example.",
            "Crédito rápido sin requisitos hasta {amount}. Solicítalo hoy en dinero-facil.example.",
            "Recibe un REGALO GRATIS. Envía el código {code} al {phone} para activarlo.",
            "Tienes una compensación de {amount}. Paga una pequeña comisión para recibirla.",
            "Trabajo desde casa sin experiencia, gana {amount} por semana. Responde ahora.",
            "Último aviso: tu cuenta será eliminada. Inicia sesión en verificar-cuenta.example.",
            "Oferta exclusiva: 90% de descuento solo durante 10 minutos. Haz clic ahora.",
            "Has recibido un bono. Comparte el código SMS con el operador del {phone}.",
        ],
        es_values,
    )

    rows = []
    for language, label, messages in [("ru", "ham", ru_ham), ("ru", "spam", ru_spam), ("es", "ham", es_ham), ("es", "spam", es_spam)]:
        rows.extend({"message": message, "label": label, "language": language, "source": "synthetic"} for message in messages)
    return pd.DataFrame(rows)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not RU_DATA_PATH.exists():
        raise FileNotFoundError(f"Russian dataset not found: {RU_DATA_PATH}")
    english = pd.read_csv(DATA_PATH, sep="\t", names=["label", "message"], encoding="utf-8")
    english["language"] = "en"
    english["source"] = "UCI"
    russian = pd.read_csv(RU_DATA_PATH, usecols=["text", "is_spam"])
    russian = russian.dropna(subset=["text", "is_spam"])
    russian = russian.loc[russian["text"].astype(str).str.len().between(3, 1_000)]
    # A fixed, class-aware subset keeps training fast enough for a live classroom demo.
    russian = pd.concat(
        [
            group.sample(n=min(len(group), 12_000 if label == 0 else 8_000), random_state=42)
            for label, group in russian.groupby("is_spam")
        ],
        ignore_index=True,
    )
    russian = russian.rename(columns={"text": "message"})
    russian["label"] = russian["is_spam"].map({0: "ham", 1: "spam"})
    russian = russian.drop(columns="is_spam").dropna(subset=["message", "label"])
    russian["language"] = "ru"
    russian["source"] = "DmitryKRX/anti_spam_ru"
    spanish = build_synthetic_multilingual_data()
    spanish = spanish.loc[spanish["language"].eq("es")]
    data = pd.concat([english, russian, spanish], ignore_index=True)
    data["normalised"] = data["message"].map(_normalise)
    data = data.drop_duplicates(subset=["normalised", "label"]).drop(columns="normalised")
    return data.reset_index(drop=True)


def detect_language(text: str) -> str:
    lowered = text.lower()
    if re.search(r"[а-яё]", lowered):
        return "ru"
    spanish_markers = ("¿", "¡", "ñ", "á", "é", "í", "ó", "ú", " has ", " tu ", " para ", " ahora")
    if any(marker in f" {lowered} " for marker in spanish_markers):
        return "es"
    return "en"


def _scores(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label="spam", zero_division=0),
    }


def train_model() -> ModelBundle:
    data = load_dataset()
    stratify_key = data["language"] + "_" + data["label"]
    train_index, test_index = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42, stratify=stratify_key)
    train, test = data.iloc[train_index], data.iloc[test_index]
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=18_000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=18_000, sublinear_tf=True)),
        ]
    )
    model = Pipeline(
        [("features", features), ("classifier", LogisticRegression(max_iter=700, class_weight="balanced", solver="liblinear", random_state=42))]
    )
    sample_weight = np.where(train["language"].eq("es"), 5.0, 1.0)
    model.fit(train["message"], train["label"], classifier__sample_weight=sample_weight)
    predictions = model.predict(test["message"])
    metrics: dict = {
        **_scores(test["label"], predictions),
        "matrix": confusion_matrix(test["label"], predictions, labels=["ham", "spam"]),
        "train_size": len(train),
        "test_size": len(test),
        "deduplicated_size": len(data),
        "by_language": {},
    }
    for language in LANGUAGE_NAMES:
        mask = test["language"].eq(language)
        metrics["by_language"][language] = _scores(test.loc[mask, "label"], predictions[mask])
    test_results = test[["message", "label", "language"]].copy()
    test_results["prediction"] = predictions
    test_results["spam_probability"] = model.predict_proba(test["message"])[:, list(model.classes_).index("spam")]
    return ModelBundle(model=model, metrics=metrics, dataset=data, test_results=test_results)


def predict_messages(bundle: ModelBundle, messages: list[str], threshold: float = 0.5) -> pd.DataFrame:
    probabilities = bundle.model.predict_proba(messages)[:, list(bundle.model.classes_).index("spam")]
    return pd.DataFrame(
        {
            "message": messages,
            "language": [detect_language(message) for message in messages],
            "spam_probability": probabilities,
            "prediction": np.where(probabilities >= threshold, "spam", "ham"),
        }
    )


def explain_message(bundle: ModelBundle, message: str, limit: int = 7) -> list[tuple[str, float]]:
    features = bundle.model.named_steps["features"]
    classifier = bundle.model.named_steps["classifier"]
    vector = features.transform([message])
    names = features.get_feature_names_out()
    contributions = vector.toarray()[0] * classifier.coef_[0]
    active = []
    for index in vector.nonzero()[1]:
        if contributions[index] > 0 and names[index].startswith("word__"):
            active.append((names[index].removeprefix("word__"), float(contributions[index])))
    return sorted(active, key=lambda item: item[1], reverse=True)[:limit]


def risk_signals(message: str) -> list[str]:
    lowered = message.lower()
    signals = []
    if re.search(r"(?:https?://|www\.|\b[a-z0-9-]+\.(?:ru|com|net|org|example)\b)", lowered):
        signals.append("link")
    if len(re.sub(r"\D", "", message)) >= 8:
        signals.append("phone")
    if re.search(r"[$€£₽]|\b(?:руб|рублей|dollars?|euros?|pounds?)\b", lowered):
        signals.append("money")
    if any(word in lowered for word in ["urgent", "срочно", "немедленно", "ahora", "urgente", "last chance", "последн"]):
        signals.append("urgency")
    if any(word in lowered for word in ["free", "бесплат", "gratis", "выиграл", "won", "ganado", "prize", "приз", "premio"]):
        signals.append("reward")
    letters = [char for char in message if char.isalpha()]
    if letters and sum(char.isupper() for char in letters) / len(letters) > 0.35:
        signals.append("caps")
    if message.count("!") >= 2:
        signals.append("exclamations")
    return signals
