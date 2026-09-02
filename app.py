from __future__ import annotations

import html
import io
from datetime import datetime

import pandas as pd
import streamlit as st

from spam_model import explain_message, predict_messages, risk_signals, train_model
from ui_text import SIGNALS, TEXT


st.set_page_config(page_title="SpamGuard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
<style>
:root {--ink:#171a2e;--muted:#697087;--primary:#5c64f4;}
.stApp {background:radial-gradient(circle at 5% 2%,rgba(92,100,244,.16),transparent 28rem),radial-gradient(circle at 96% 7%,rgba(22,190,155,.12),transparent 25rem),#f7f8fc;}
header[data-testid="stHeader"],#MainMenu,[data-testid="stToolbar"],[data-testid="stDecoration"],footer {display:none!important;}
[data-testid="stElementToolbar"] {display:none!important;}
[data-testid="stSidebar"] {background:rgba(248,249,253,.94);border-right:1px solid #e6e8f1;}
.block-container {max-width:1180px;padding-top:2.3rem;padding-bottom:4rem;} h1,h2,h3 {letter-spacing:-.028em;color:var(--ink);}
.brand {display:flex;align-items:center;gap:.65rem;font-weight:850;font-size:1.05rem;color:#2c3149;}
.brand-icon {display:grid;place-items:center;width:2.3rem;height:2.3rem;border-radius:12px;background:#5963f3;color:white;box-shadow:0 8px 22px rgba(76,86,225,.28);}
.brand small {color:#858ba1;font-weight:600;} .hero {padding:1.6rem 0 1.15rem;}
.hero h1 {font-size:clamp(2.5rem,5.3vw,4.8rem);line-height:1.01;margin:.4rem 0 1rem;} .hero h1 span {color:var(--primary);}
.hero p {font-size:1.08rem;line-height:1.65;color:var(--muted);max-width:760px;}
.result {padding:1.45rem;border-radius:22px;border:1px solid;margin:.6rem 0 1rem;} .result.spam {background:#fff1f3;border-color:#ffcbd1;color:#952633;}
.result.ham {background:#ebfcf6;border-color:#b4edd9;color:#12624b;} .eyebrow {font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;font-weight:850;opacity:.68;}
.result-title {font-size:1.65rem;font-weight:850;margin:.3rem 0 .2rem;}
.pill {display:inline-block;padding:.32rem .68rem;margin:.18rem .22rem .18rem 0;background:#edf0ff;color:#414bc4;border-radius:999px;font-size:.85rem;font-weight:700;}
.signal {display:inline-block;padding:.31rem .62rem;margin:.16rem .2rem .16rem 0;background:#fff4dd;color:#8a5a09;border:1px solid #f6dfac;border-radius:9px;font-size:.84rem;font-weight:650;}
.tiny {color:#7a8095;font-size:.84rem;line-height:1.55;} .phone {max-width:560px;margin:auto;padding:12px;border-radius:32px;background:#22263a;box-shadow:0 22px 60px rgba(23,26,46,.2);}
.phone-inner {padding:1rem;border-radius:23px;background:white;} .history-row {padding:.75rem 0;border-bottom:1px solid #eceef4;} .history-row:last-child {border-bottom:0;}
.step {height:100%;padding:1.05rem;border:1px solid #e4e7f0;border-radius:18px;background:white;} .step-num {display:grid;place-items:center;width:2rem;height:2rem;border-radius:10px;background:#eef0ff;color:#4e57d8;font-weight:850;margin-bottom:.7rem;}
div[data-testid="stMetric"] {background:white;border:1px solid #e5e8f0;padding:1rem;border-radius:18px;box-shadow:0 9px 30px rgba(37,42,82,.05);}
.stButton>button,.stDownloadButton>button {border-radius:13px;min-height:2.8rem;font-weight:750;}
</style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_bundle():
    return train_model()


bundle = get_bundle()
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {"yes": 0, "no": 0}

language_option = st.sidebar.selectbox("Язык интерфейса", ["Русский", "English", "Español"])
ui_language = {"Русский": "ru", "English": "en", "Español": "es"}[language_option]
t = TEXT[ui_language]
threshold = st.sidebar.slider(t["sensitivity"], 0.20, 0.80, 0.50, 0.05, help=t["sensitivity_help"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(bundle.dataset):,}** {t['messages']}".replace(",", " "))
st.sidebar.caption(t["supported_languages"])
st.sidebar.caption(t["model_name"])
if st.sidebar.button(t["clear"], width="stretch"):
    st.session_state.history = []

st.markdown(f'<div class="brand"><span class="brand-icon">S</span>SpamGuard <small>/ {t["tagline"]}</small></div>', unsafe_allow_html=True)
st.markdown(f'<section class="hero"><h1>{t["hero"]}</h1><p>{t["subtitle"]}</p></section>', unsafe_allow_html=True)

examples = {
    "—": "", "🇷🇺 Обычное": "Маша, встречаемся у метро в семь? Я немного задержусь.",
    "🇷🇺 Спам": "СРОЧНО! Вы выиграли 100 000 рублей! Перейдите по ссылке prize-now.example и заберите приз!",
    "🇬🇧 Regular": "Are we still meeting near the station at 7 tonight?",
    "🇬🇧 Spam": "WINNER! You won a £1000 cash prize. Click claim-now.example immediately!",
    "🇪🇸 Normal": "Ana, ¿nos vemos en la estación esta tarde?",
    "🇪🇸 Spam": "¡URGENTE! Has ganado 1.000 €. Reclama tu premio en regalo.example ahora.",
}
check_tab, batch_tab, analytics_tab, method_tab = st.tabs(t["tabs"])

with check_tab:
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        selected = st.selectbox(t["example"], list(examples))
        message = st.text_area("Message", value=examples[selected], height=180, placeholder=t["placeholder"], label_visibility="collapsed")
        analyse = st.button(t["analyze"], type="primary", width="stretch")
        st.markdown(f'<div class="tiny">🔒 {t["privacy"]}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="phone"><div class="phone-inner">', unsafe_allow_html=True)
        st.caption(f"SpamGuard · {t['preview']}")
        if not st.session_state.history:
            st.markdown("#### 0–100%")
            st.caption(t["sensitivity_help"])
            st.progress(0)
        else:
            last = st.session_state.history[0]
            st.markdown(f"#### {last['probability']:.0%}")
            st.progress(last["probability"])
            st.caption(last["message"][:105] + ("…" if len(last["message"]) > 105 else ""))
        st.markdown("</div></div>", unsafe_allow_html=True)

    if analyse:
        clean_message = message.strip()
        if not clean_message:
            st.warning(t["empty"])
        else:
            prediction = predict_messages(bundle, [clean_message], threshold).iloc[0]
            probability = float(prediction.spam_probability)
            is_spam = prediction.prediction == "spam"
            st.session_state.history.insert(0, {"message": clean_message, "probability": probability, "prediction": prediction.prediction, "language": prediction.language, "time": datetime.now().strftime("%H:%M")})
            st.session_state.history = st.session_state.history[:8]
            st.markdown("---")
            result_col, score_col = st.columns([1.45, .55], gap="large")
            with result_col:
                css_class, icon = ("spam", "⚠") if is_spam else ("ham", "✓")
                title, advice = (t["spam"], t["spam_advice"]) if is_spam else (t["ham"], t["ham_advice"])
                st.markdown(f'<div class="result {css_class}"><div class="eyebrow">{t["result"]}</div><div class="result-title">{icon} {title}</div><div>{advice}</div></div>', unsafe_allow_html=True)
            with score_col:
                st.metric(t["risk"], f"{probability:.0%}")
                st.metric(t["detected"], t["language_names"][prediction.language])
            words, signals = explain_message(bundle, clean_message), risk_signals(clean_message)
            details_left, details_right = st.columns(2, gap="large")
            with details_left:
                st.markdown(f"**{t['influence']}**")
                st.markdown(" ".join(f'<span class="pill">{html.escape(term)}</span>' for term, _ in words) if words else "—", unsafe_allow_html=True)
            with details_right:
                st.markdown(f"**{t['signals']}**")
                st.markdown(" ".join(f'<span class="signal">{SIGNALS[ui_language][key]}</span>' for key in signals) if signals else "—", unsafe_allow_html=True)
            st.markdown(f"**{t['feedback']}**")
            feedback_cols = st.columns([.14, .14, .72])
            if feedback_cols[0].button(f"👍 {t['yes']}"):
                st.session_state.feedback["yes"] += 1
            if feedback_cols[1].button(f"👎 {t['no']}"):
                st.session_state.feedback["no"] += 1

    if st.session_state.history:
        st.markdown(f"### {t['history']}")
        for item in st.session_state.history[:5]:
            badge = "🔴" if item["prediction"] == "spam" else "🟢"
            st.markdown(f'<div class="history-row">{badge} <b>{item["probability"]:.0%}</b> · {html.escape(item["message"][:95])} <span class="tiny"> · {item["time"]}</span></div>', unsafe_allow_html=True)

with batch_tab:
    st.subheader(t["batch_title"])
    st.caption(t["batch_help"])
    uploaded = st.file_uploader(t["upload"], type=["csv", "txt"])
    batch_data = None
    if uploaded is not None:
        raw = uploaded.getvalue()
        if uploaded.name.lower().endswith(".txt"):
            batch_data = pd.DataFrame({"message": [line.strip() for line in raw.decode("utf-8", errors="replace").splitlines() if line.strip()]})
            text_column = "message"
        else:
            try:
                batch_data = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
            except UnicodeDecodeError:
                batch_data = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="cp1251")
            text_column = st.selectbox(t["column"], list(batch_data.columns))
        if batch_data is None or batch_data.empty:
            st.warning(t["batch_empty"])
        elif st.button(t["run_batch"], type="primary"):
            messages = batch_data[text_column].fillna("").astype(str).head(1000).tolist()
            result = predict_messages(bundle, messages, threshold)
            result["spam_probability"] = result["spam_probability"].round(4)
            spam_count = int(result.prediction.eq("spam").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric(t["batch_messages"], len(result)); m2.metric(t["found"], spam_count); m3.metric(t["risk"], f"{result.spam_probability.mean():.0%}")
            display_result = result.sort_values("spam_probability", ascending=False).copy()
            display_result["language"] = display_result["language"].map(t["language_names"])
            display_result["prediction"] = display_result["prediction"].map(t["class_names"])
            st.dataframe(display_result.rename(columns=t["batch_columns"]), width="stretch", hide_index=True)
            st.download_button(t["download"], result.to_csv(index=False).encode("utf-8-sig"), "spamguard_results.csv", "text/csv")

with analytics_tab:
    metrics = bundle.metrics
    st.subheader(t["analytics_title"])
    st.caption(f"{t['analytics_caption']} · {t['test_label']}: {metrics['test_size']} · {t['train_label']}: {metrics['train_size']}")
    metric_cols = st.columns(4)
    for column, (label, key) in zip(metric_cols, zip(t["metric_names"], ["accuracy", "precision", "recall", "f1"])):
        column.metric(label, f"{metrics[key]:.1%}")
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.markdown(f"#### {t['languages']}")
        language_metrics = pd.DataFrame(metrics["by_language"]).T.reset_index(names="language")
        language_metrics["language"] = language_metrics["language"].map(t["language_names"])
        language_metrics = language_metrics.rename(columns=t["chart_metric_names"])
        st.bar_chart(language_metrics.set_index("language")[list(t["chart_metric_names"].values())], horizontal=True)
    with chart_right:
        st.markdown(f"#### {t['dataset']}")
        composition = bundle.dataset.groupby(["language", "label"]).size().unstack(fill_value=0)
        composition.index = composition.index.map(t["language_names"])
        composition = composition.rename(columns=t["class_names"])
        st.bar_chart(composition, horizontal=True)
    st.markdown(f"#### {t['matrix']}")
    matrix = pd.DataFrame(metrics["matrix"], index=t["matrix_rows"], columns=t["matrix_columns"])
    st.dataframe(matrix, width="stretch")
    st.caption(t["metric_caption"])

with method_tab:
    st.subheader(t["method_title"])
    steps = [(f"{index:02}", title, body) for index, (title, body) in enumerate(t["method_steps"], start=1)]
    columns = st.columns(4)
    for column, (number, title, body) in zip(columns, steps):
        column.markdown(f'<div class="step"><div class="step-num">{number}</div><b>{title}</b><div class="tiny" style="margin-top:.45rem">{body}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(t["method_body"])
    st.markdown(f"### {t['limits']}")
    st.warning(t["limits_body"])
    st.caption(t.get("sample_note", "Для быстрого запуска используется фиксированная выборка русских сообщений."))
    st.markdown(f"**{t['sources']}**")
