import os
import textwrap
from datetime import datetime
import json

import streamlit as st
from huggingface_hub import InferenceClient

import gspread
from google.oauth2.service_account import Credentials


# ---------------------------
# CONFIG
# ---------------------------

# Choose a chat / instruction model that supports Russian reasonably well.
# You can change this to another instruct model if needed.
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Spreadsheet constants
SPREADSHEET_NAME = "AI_Campaign_Control"
JOBPOSTS_SHEET = "JobPosts"
RESEARCH_SHEET = "ResearchInsights"

# Read secrets
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
GCP_SERVICE_ACCOUNT = st.secrets.get("gcp_service_account", None)


# ---------------------------
# HELPERS: Hugging Face client
# ---------------------------

@st.cache_resource(show_spinner=False)
def get_hf_client():
    """Create Hugging Face Inference client if token is available."""
    if not HF_TOKEN:
        return None
    try:
        client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
        return client
    except Exception:
        return None


def call_model(system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
    client = get_hf_client()
    if client is None:
        return "⚠️ Модель не настроена. Проверьте токен HF_TOKEN."

    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
        )

        return response.choices[0].message["content"]

    except Exception as e:
        return f"⚠️ Ошибка при вызове модели: {e}"



# ---------------------------
# HELPERS: Google Sheets
# ---------------------------

@st.cache_resource(show_spinner=False)
def get_gsheet_client():
    """Create gspread client from service account info in secrets. Returns None if not configured."""
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
        st.success("✅ Результаты исследования сохранены в Google Sheets (ResearchInsights).")
    except Exception as e:
        st.error(f"Ошибка при записи в Google Sheets: {e}")


# ---------------------------
# STREAMLIT UI
# ---------------------------

st.set_page_config(
    page_title="AI Campaign Assistant",
    layout="wide",
)

st.title("AI Campaign Assistant")
st.caption(
    "Генерация вакансий и аналитика кампаний с помощью онлайн-модели (Hugging Face) "
    "и интеграции с Google Sheets."
)

mode = st.sidebar.radio(
    "Режим работы",
    ["📝 Вакансии и резюме (Posts & Summaries)", "📊 Аналитика и исследования (Research & Insights)"],
)

st.sidebar.markdown("---")
st.sidebar.write(f"Модель: `{MODEL_ID}`")

if not HF_TOKEN:
    st.sidebar.error("HF_TOKEN не настроен в secrets. Модель работать не будет.")
if not GCP_SERVICE_ACCOUNT:
    st.sidebar.warning("gcp_service_account не настроен — запись в Google Sheets отключена.")


# ---------------------------
# MODE 1: POSTS & SUMMARIES (RUSSIAN JOB POSTS)
# ---------------------------
if mode.startswith("📝"):
    st.subheader("📝 Генерация вакансий и кратких резюме (по-русски)")

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
        variant_label = st.selectbox(
            "Вариант (A/B/C)", ["A", "B", "C"], index=0
        )
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

    generated_post = None
    summary_text = None

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
                    Составь русскоязычное объявление о вакансии в стиле ниже (с эмодзи и структурой):

                    ПРИМЕР СТИЛЯ:
                    "2025.0021 – Установщик кухонь

                    👤 Должность: Установщик кухонь – 3 вакансии
                    💶 Оплата (чистыми): 15,50 € / час
                    📅 График / период работы: Пн–Пт, с 08:00. 180–220 часов в месяц.
                    🦺 Рабочая одежда: Предоставляется.
                    🔧 Инструменты: Предоставляются бесплатно.
                    🚙 Транспорт до работы: Бесплатно (служебный автомобиль).

                    📍 Требования / Для кого:
                    Мужчины 25–45 лет.
                    Опыт работы обязателен – от 1 года.
                    Навыки сборки и установки мебели, подключения бытовой техники.
                    Знание языка: немецкий на уровне A2 (для общения с клиентами).

                    📝 Необходимые документы:
                    Паспорт ЕС, Параграф 24, водительское удостоверение категории B (преимущество).

                    📋 Обязанности:
                    Доставка и подъем кухонных гарнитуров
                    Сборка, установка и выравнивание модулей
                    Врезка и монтаж моек, варочных панелей
                    Подключение бытовой техники

                    🧾 Испытательный срок: 5 рабочих дней
                    📆 Начало работы: Срочно

                    👉 Заинтересованы?
                    Заполните форму и получите работу: <ссылка>"

                    ТЕПЕРЬ СДЕЛАЙ НОВУЮ ВАКАНСИЮ ПО ЭТИМ ДАННЫМ:

                    Должность: {job_title}
                    Город / Регион: {city}
                    Целевая аудитория: {target_audience}
                    Вариант: {variant_label}
                    Платформа: {platform}
                    Предпочтительный тон: {tone}

                    Сырой текст / заметки:
                    {raw_description}

                    Требования к результату:
                    - Сохрани похожую структуру и эмодзи-блоки, как в примере.
                    - Обязательно укажи город/регион.
                    - Если есть информация о зарплате, графике, жилье, транспорте — выдели её.
                    - Текст должен быть понятен русскоязычным рабочим.

                    Пиши текст на русском языке.
                    {cta}
                    """
                )

                generated_post = call_model(system_prompt, user_prompt, max_new_tokens=350)

            st.markdown("#### ✏️ Сгенерированное объявление (русский)")
            st.write(generated_post)

            if st.button("💾 Сохранить объявление в Google Sheets"):
                if generated_post:
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
                else:
                    st.warning("Сначала сгенерируйте объявление.")

    # --- Generate employer-facing summary ---
    if gen_col2.button("📄 Сгенерировать краткое резюме для работодателя (ENG/RU)"):
        if not job_title or not city or not raw_description:
            st.error("Пожалуйста, заполните хотя бы Должность, Город и Сырой текст.")
        else:
            with st.spinner("Генерирую краткое резюме..."):
                system_prompt = (
                    "You create concise professional summaries for internal use by employers "
                    "and project managers. You highlight key points and avoid marketing fluff. "
                    "You can mix Russian and English if needed."
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

            st.markdown("#### 📄 Краткое резюме для работодателя")
            st.write(summary_text)


# ---------------------------
# MODE 2: RESEARCH & INSIGHTS
# ---------------------------
else:
    st.subheader("📊 Аналитика и исследования (Research & Insights)")

    st.markdown(
        "Здесь можно задавать вопросы про стратегии, платформы, регионы, "
        "или вставлять свои KPI и просить пояснения."
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
                        "Ты — аналитик по маркетингу и рекрутингу для blue-collar платформы. "
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


