import os
import textwrap
from datetime import datetime

import streamlit as st
from huggingface_hub import InferenceClient

import gspread
from google.oauth2.service_account import Credentials


# ---------------------------
# CONFIG
# ---------------------------

# Hugging Face model (supports chat / conversational)
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Google Sheets
SPREADSHEET_NAME = "AI_Campaign_Control"
JOBPOSTS_SHEET = "JobPosts"
RESEARCH_SHEET = "ResearchInsights"

# Secrets (set in Streamlit Cloud → Settings → Secrets)
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
GCP_SERVICE_ACCOUNT = st.secrets.get("gcp_service_account", None)


# ---------------------------
# HUGGING FACE CLIENT (CHAT COMPLETION)
# ---------------------------

@st.cache_resource(show_spinner=False)
def get_hf_client():
    """Create HF InferenceClient if token is set."""
    if not HF_TOKEN:
        return None
    try:
        return InferenceClient(MODEL_ID, token=HF_TOKEN)
    except Exception:
        return None


def call_model(system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
    """
    Call the HF chat model using chat_completion.

    This fixes the previous error:
    'Model ... is not supported for task text-generation. Supported task: conversational.'
    """
    client = get_hf_client()
    if client is None:
        return "⚠️ Модель не настроена. Проверьте HF_TOKEN в secrets."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
        )
        # HF chat_completion returns an object with choices
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"⚠️ Ошибка при вызове модели: {e}"


# ---------------------------
# GOOGLE SHEETS HELPERS
# ---------------------------

@st.cache_resource(show_spinner=False)
def get_gsheet_client():
    """Create gspread client from service-account info in secrets."""
    if not GCP_SERVICE_ACCOUNT:
        return None

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        credentials = Credentials.from_service_account_info(
            GCP_SERVICE_ACCOUNT, scopes=scopes
        )
        client = gspread.authorize(credentials)
        return client
    except Exception:
        return None


def append_jobpost_to_sheet(
    timestamp: datetime,
    job_title: str,
    city: str,
    platform: str,
    variant_label: str,
    target_audience: str,
    application_link: str,
    generated_post: str,
):
    gc = get_gsheet_client()
    if gc is None:
        st.warning(
            "Google Sheets не настроен. Проверьте gcp_service_account в secrets и доступ к таблице."
        )
        return

    try:
        sh = gc.open(SPREADSHEET_NAME)
        try:
            ws = sh.worksheet(JOBPOSTS_SHEET)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=JOBPOSTS_SHEET, rows=1000, cols=10)

        ws.append_row(
            [
                timestamp.isoformat(),
                job_title,
                city,
                platform,
                variant_label,
                target_audience,
                application_link,
                generated_post,
            ]
        )
        st.success("✅ Объявление сохранено в Google Sheets (JobPosts).")
    except Exception as e:
        st.error(f"Ошибка при записи в Google Sheets: {e}")


def append_research_to_sheet(
    timestamp: datetime,
    question_type: str,
    input_text: str,
    insights: str,
):
    gc = get_gsheet_client()
    if gc is None:
        st.warning(
            "Google Sheets не настроен. Проверьте gcp_service_account в secrets и доступ к таблице."
        )
        return

    try:
        sh = gc.open(SPREADSHEET_NAME)
        try:
            ws = sh.worksheet(RESEARCH_SHEET)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=RESEARCH_SHEET, rows=1000, cols=10)

        ws.append_row(
            [
                timestamp.isoformat(),
                question_type,
                input_text,
                insights,
            ]
        )
        st.success("✅ Инсайты сохранены в Google Sheets (ResearchInsights).")
    except Exception as e:
        st.error(f"Ошибка при записи в Google Sheets: {e}")


# ---------------------------
# UI CONFIG
# ---------------------------

st.set_page_config(
    page_title="AI Campaign Assistant",
    layout="wide",
)

st.title("AI Campaign Assistant")
st.caption(
    "Внутренний инструмент для генерации русскоязычных вакансий и аналитики кампаний. "
    "Основан на онлайн-модели (Hugging Face) с интеграцией в Google Sheets."
)

# Sidebar status
st.sidebar.header("Статус системы")

st.sidebar.write(f"Модель: `{MODEL_ID}`")

if HF_TOKEN:
    st.sidebar.success("HF_TOKEN настроен ✅")
else:
    st.sidebar.error("HF_TOKEN не настроен ❌ — модель не будет работать.")

if GCP_SERVICE_ACCOUNT:
    st.sidebar.success("Google Sheets интеграция включена ✅")
else:
    st.sidebar.warning("gcp_service_account не настроен — запись в Sheets отключена.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "1. Вкладка **Вакансии** — генерируем объявления и резюме.\n"
    "2. Вкладка **Аналитика** — задаём вопросы, интерпретируем метрики."
)

# Tabs layout
tab_posts, tab_research = st.tabs(
    ["📝 Вакансии и краткие резюме", "📊 Аналитика и исследования"]
)


# ---------------------------
# TAB 1: POSTS & SUMMARIES
# ---------------------------
with tab_posts:
    st.subheader("📝 Генерация вакансий и кратких резюме (по-русски)")

    st.info(
        "1. Заполните поля.\n"
        "2. Нажмите **'Сгенерировать объявление на русском'**.\n"
        "3. При необходимости сохраните результат в Google Sheets."
    )

    col1, col2 = st.columns(2)

    with col1:
        job_title = st.text_input("Должность / Job Title", placeholder="Установщик кухонь")
        city = st.text_input("Город / Регион", placeholder="Гамбург")
        platform = st.selectbox(
            "Платформа",
            ["Facebook", "Instagram", "Telegram", "WhatsApp", "Generic"],
            index=0,
        )
        tone = st.selectbox(
            "Тон объявления",
            ["Простой и понятный", "Дружелюбный", "Профессиональный", "Срочно, но без паники"],
            index=0,
        )

    with col2:
        target_audience = st.text_input(
            "Целевая аудитория",
            placeholder="Иммигранты, ищущие работу в Германии в сфере монтажа кухонь",
        )
        variant_label = st.selectbox("Вариант (A/B/C)", ["A", "B", "C"], index=0)
        application_link = st.text_input(
            "Ссылка на форму / анкету (опционально)",
            placeholder="https://docs.google.com/forms/...",
        )

    st.markdown("#### Сырой текст / заметки по вакансии")
    raw_description = st.text_area(
        "Опишите детали: зарплата, график, обязанности, требования, документы, жилье и т.д.",
        height=220,
        placeholder="Сюда можно вставить пример вроде твоего 2025.0021 – Установщик кухонь...",
    )

    gen_col1, gen_col2 = st.columns(2)

    generated_post = st.session_state.get("generated_post", "")
    summary_text = st.session_state.get("summary_text", "")

    # --- Generate Russian job post ---
    if gen_col1.button("✏️ Сгенерировать объявление на русском"):
        if not job_title or not city or not raw_description:
            st.error("Пожалуйста, заполните хотя бы Должность, Город и Сырой текст.")
        else:
            with st.spinner("Генерирую объявление..."):
                system_prompt = (
                    "Ты — помощник по созданию вакансий для рабочих (blue-collar) в Германии. "
                    "Ты пишешь объявления в дружелюбном, понятном стиле на РУССКОМ языке. "
                    "Используй эмодзи для структурирования текста, как в объявлениях в Telegram/WhatsApp.\n\n"
                    "Правила:\n"
                    "- Пиши коротко и по делу, без лишней рекламы.\n"
                    "- Делай чёткие блоки: должность, оплата, график, требования, документы, обязанности, начало работы.\n"
                    "- В конце всегда добавляй понятный призыв к действию (заполнить форму / написать в WhatsApp).\n"
                    "- Сохраняй профессиональный, но простой стиль для русскоязычных работников."
                )

                if application_link.strip():
                    cta = (
                        f"В конце добавь блок с призывом:\n"
                        f"👉 Заинтересованы? Заполните форму по ссылке: {application_link}\n"
                    )
                else:
                    cta = (
                        "В конце добавь блок с призывом:\n"
                        "👉 Заинтересованы? Напишите нам в WhatsApp или заполните анкету.\n"
                    )

                user_prompt = textwrap.dedent(
                    f"""
                    Составь русскоязычное объявление о вакансии в стиле ниже (с эмодзи и структурой).

                    ПРИМЕР СТИЛЯ:
                    "2025.0021 – Установщик кухонь
                    👤 Должность: Установщик кухонь – 3 вакансии
                    💶 Оплата (чистыми): 15,50 € / час
                    📅 График / период работы: Пн–Пт, с 08:00. 180–220 часов в месяц.
                    🦺 Рабочая одежда: Предоставляется.
                    🔧 Инструменты: Предоставляются бесплатно.
                    🚙 Транспорт до работы: Бесплатно (служебный автомобиль).
                    ..."

                    ТЕПЕРЬ СДЕЛАЙ НОВУЮ ВАКАНСИЮ ПО ЭТИМ ДАННЫМ:

                    Должность: {job_title}
                    Город / Регион: {city}
                    Целевая аудитория: {target_audience}
                    Вариант: {variant_label}
                    Платформа: {platform}
                    Предпочтительный тон: {tone}

                    Сырой текст / заметки:
                    {raw_description}

                    Требования:
                    - Сохрани похожую структуру и эмодзи-блоки, как в примере.
                    - Обязательно укажи город/регион.
                    - Если есть информация о зарплате, графике, жилье, транспорте — выдели её.
                    - Текст должен быть понятен русскоязычным рабочим.

                    Пиши текст на русском языке.
                    {cta}
                    """
                )

                generated_post = call_model(system_prompt, user_prompt, max_new_tokens=350)
                st.session_state["generated_post"] = generated_post

    if generated_post:
        st.markdown("#### ✏️ Сгенерированное объявление (русский)")
        st.write(generated_post)

        if st.button("💾 Сохранить объявление в Google Sheets"):
            append_jobpost_to_sheet(
                timestamp=datetime.utcnow(),
                job_title=job_title,
                city=city,
                platform=platform,
                variant_label=variant_label,
                target_audience=target_audience,
                application_link=application_link,
                generated_post=generated_post,
            )

    # --- Generate employer-facing summary ---
    if gen_col2.button("📄 Сгенерировать краткое резюме для работодателя (ENG/RU)"):
        if not job_title or not city or not raw_description:
            st.error("Пожалуйста, заполните хотя бы Должность, Город и Сырой текст.")
        else:
            with st.spinner("Генерирую краткое резюме..."):
                system_prompt = (
                    "You create concise professional summaries for internal use by employers "
                    "and project managers. You highlight key points and avoid marketing fluff. "
                    "You may mix Russian and English if helpful."
                )

                user_prompt = textwrap.dedent(
                    f"""
                    Summarize this job in 5–7 bullet points for an internal report.
                    Focus on:
                    - job title
                    - location
                    - key requirements
                    - salary/benefits (if mentioned)
                    - ideal candidate profile

                    Job title: {job_title}
                    City / Region: {city}

                    Raw details:
                    {raw_description}
                    """
                )

                summary_text = call_model(system_prompt, user_prompt, max_new_tokens=250)
                st.session_state["summary_text"] = summary_text

    if summary_text:
        st.markdown("#### 📄 Краткое резюме для работодателя")
        st.write(summary_text)


# ---------------------------
# TAB 2: RESEARCH & INSIGHTS
# ---------------------------
with tab_research:
    st.subheader("📊 Аналитика и исследования")

    st.info(
        "Сюда можно вставить результаты кампаний, KPI-таблицы или просто задать вопрос "
        "про регионы и платформы. AI вернёт краткий анализ и рекомендации."
    )

    research_type = st.selectbox(
        "Тип запроса",
        [
            "Интерпретация метрик / таблиц",
            "Сравнение регионов / платформ",
            "Общий вопрос по стратегии / трафику",
        ],
    )

    st.markdown("#### Ваш вопрос или данные")
    research_input = st.text_area(
        "Опишите, что нужно. Можно вставить таблицу (копипаст), текст с результатами или задать вопрос.",
        height=240,
        placeholder="Примеры:\n"
        "- 'Вот результаты по вариантам A/B/C по городам — что лучше работает и почему?'\n"
        "- 'Какие платформы лучше для вакансий складских работников в Гамбурге vs Берлине?'\n"
        "- 'У нас мало откликов из Киля, что можно изменить?'",
    )

    if st.button("🔍 Получить инсайты"):
        if not research_input.strip():
            st.error("Пожалуйста, введите вопрос или данные.")
        else:
            with st.spinner("Анализирую..."):
                if research_type == "Интерпретация метрик / таблиц":
                    system_prompt = (
                        "Ты — аналитик по маркетингу и рекрутингу для платформы по найму рабочих. "
                        "Ты интерпретируешь KPI, варианты объявлений и результаты по регионам. "
                        "Отвечай кратко и по-русски, давай только практические выводы."
                    )
                elif research_type == "Сравнение регионов / платформ":
                    system_prompt = (
                        "Ты — эксперт по каналам трафика и географии. "
                        "Сравниваешь города/регионы и платформы (Facebook, Instagram, Telegram, WhatsApp, job boards) "
                        "с точки зрения трафика и откликов. Отвечай по-русски."
                    )
                else:
                    system_prompt = (
                        "Ты — стратег по перформанс-маркетингу в рекрутинге. "
                        "Отвечай по-русски, давай конкретные шаги и рекомендации."
                    )

                user_prompt = textwrap.dedent(
                    f"""
                    Вот мой вопрос / данные:

                    {research_input}

                    Пожалуйста:
                    1) Кратко опиши, что происходит.
                    2) Выдели самые важные риски или возможности.
                    3) Дай 3–5 конкретных, практических рекомендаций, что делать дальше.
                    """
                )

                insights = call_model(system_prompt, user_prompt, max_new_tokens=500)

            st.markdown("#### 📌 Инсайты и рекомендации")
            st.write(insights)

            if st.button("💾 Сохранить инсайты в Google Sheets"):
                append_research_to_sheet(
                    timestamp=datetime.utcnow(),
                    question_type=research_type,
                    input_text=research_input,
                    insights=insights,
                )
