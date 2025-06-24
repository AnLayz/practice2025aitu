import os
import io
import logging
import asyncio
import textwrap
import psycopg2
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import pytesseract
import streamlit as st
import requests
from PIL import Image
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult

# ENV LOGGING 
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)8s | %(message)s")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DB_URI    = os.getenv("POSTGRES_URI", "")
BAD_WORDS = {"badword1", "badword2", "badword3"}

# PostgreSQL логирование 
def init_db():
    if not DB_URI:
        return
    try:
        with psycopg2.connect(DB_URI) as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT,
                    question TEXT,
                    answer TEXT,
                    model TEXT,
                    temperature FLOAT,
                    bad_words TEXT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
    except Exception as e:
        logging.error("Ошибка создания таблицы в PostgreSQL: %s", e)


def log_to_db(chat_id: int, question: str, answer: str, model: str, temperature: float, bad_words: List[str]):
    if not DB_URI:
        return
    try:
        with psycopg2.connect(DB_URI) as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO logs (chat_id, question, answer, model, temperature, bad_words)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (chat_id, question, answer, model, temperature, bad_words))
            conn.commit()
    except Exception as e:
        logging.error("Ошибка записи в PostgreSQL: %s", e)

#UI TEXTS
UI_TEXTS = {
    "Русский": {
        "title": "📄 PDF-Анализатор",
        "upload": "Загрузите PDF-файл",
        "question": "Введите ваш вопрос к документу:",
        "analyze": "🔍 Проанализировать",
        "bad_words": "⚠️ Запрещённые слова: ",
        "extracting": "📜 Читаем PDF…",
        "indexing": "🔬 Индексируем…",
        "answer": "✅ Ответ:",
        "download": "💾 Скачать ответ.txt",
        "model": "LLM модель",
        "creativity": "Креативность (temperature)",
        "chat_id": "Введите ваш Telegram chat ID (число):",
        "invalid_id": "❗ Сначала укажите корректный Telegram ID — только цифры.",
    },
    "English": {
        "title": "📄 PDF Analyzer",
        "upload": "Upload PDF file",
        "question": "Enter your question about the document:",
        "analyze": "🔍 Analyze",
        "bad_words": "⚠️ Forbidden words: ",
        "extracting": "📜 Reading PDF…",
        "indexing": "🔬 Indexing…",
        "answer": "✅ Answer:",
        "download": "💾 Download answer.txt",
        "model": "LLM model",
        "creativity": "Creativity (temperature)",
        "chat_id": "Enter your Telegram chat ID (number):",
        "invalid_id": "❗ Please enter a valid numeric Telegram chat ID first.",
    },
    "Қазақша": {
        "title": "📄 PDF талдаушысы",
        "upload": "PDF файлды жүктеңіз",
        "question": "Құжатқа қатысты сұрағыңызды енгізіңіз:",
        "analyze": "🔍 Талдау",
        "bad_words": "⚠️ Рұқсат етілмеген сөздер: ",
        "extracting": "📜 PDF оқылып жатыр…",
        "indexing": "🔬 Индекстеу…",
        "answer": "✅ Жауап:",
        "download": "💾 Жауапты жүктеу",
        "model": "LLM моделі",
        "creativity": "Креативтілік (temperature)",
        "chat_id": "Telegram chat ID енгізіңіз (тек сан):",
        "invalid_id": "❗ Алдымен дұрыс Telegram chat ID енгізіңіз — тек сандар.",
    }
}

#HELPERS

def send_telegram(chat_id: int, text: str, inline: bool = False) -> None:
    if not BOT_TOKEN or not chat_id:
        return
    for chunk in (text[i:i + 4000] for i in range(0, len(text), 4000)):
        try:
            req = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "Markdown",
            }
            if inline:
                req["reply_markup"] = {
                    "inline_keyboard": [[{
                        "text": "⏩ Задать новый вопрос",
                        "callback_data": "new_question"
                    }]]
                }
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", json=req, timeout=10)
        except Exception as exc:
            logging.error("Telegram error: %s", exc)


def extract_pages(uploaded) -> List[Tuple[str, int]]:
    pdf = fitz.open(stream=uploaded.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()
        if not text:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.pil_tobytes()))
            text = pytesseract.image_to_string(img, lang="rus+eng")
        pages.append((text, i))
    return pages


def chunk_pages(pages: List[Tuple[str, int]], chars_per_chunk: int = 1500) -> List[Document]:
    docs: List[Document] = []
    for text, page_num in pages:
        for chunk in textwrap.wrap(text, chars_per_chunk, break_long_words=False):
            meta = {"page": page_num}
            docs.append(Document(page_content=chunk, metadata=meta))
    return docs


def find_bad_words(text: str) -> List[str]:
    lowered = text.lower()
    return [w for w in BAD_WORDS if w in lowered]


#UI LAYOUT

def main():
    init_db()
    st.set_page_config(page_title="PDF‑Анализатор", page_icon="📄", layout="centered")

    lang = st.selectbox("UI language / Тіл / Язык", ("Русский", "English", "Қазақша"))
    txt = UI_TEXTS[lang]

    st.title(txt["title"])

    chat_id_str = st.text_input(txt["chat_id"])
    if not chat_id_str.strip().isdigit():
        st.info(txt["invalid_id"])
        st.stop()
    chat_id = int(chat_id_str.strip())

    model_name = st.selectbox(txt["model"], ("llama3.2:latest", "mistral:instruct", "phi3:mini"))
    temperature = st.slider(txt["creativity"], 0.0, 1.2, 0.2, 0.05)

    uploaded_file = st.file_uploader(txt["upload"], type=["pdf"])
    if not uploaded_file:
        st.stop()

    question = st.text_input(txt["question"])
    if not question:
        st.stop()

    if st.button(txt["analyze"]):
        bad = find_bad_words(question)
        if bad:
            warn = f"{txt['bad_words']}{', '.join(bad)}"
            st.warning(warn)
            send_telegram(chat_id, warn)

        with st.status(txt["extracting"], expanded=False):
            pages = extract_pages(uploaded_file)
            st.write(f"Извлечено страниц: {len(pages)}")

        with st.status(txt["indexing"], expanded=False):
            docs = chunk_pages(pages)
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectordb = Chroma.from_documents(docs, embeddings, persist_directory=".cache/chroma")
            retriever = vectordb.as_retriever(search_kwargs={"k": 5})
            st.write("Чанков в индексе:", len(docs))

        context_docs = retriever.get_relevant_documents(question)
        cited_context = "\n\n".join([
            f"[стр. {d.metadata['page']}]:\n{d.page_content}" for d in context_docs
        ])
        prompt_text = (
            "Используй контекст из PDF (с указанием страниц) и ответь на вопрос."
            " Если ответа нет в тексте — скажи об этом.\n\n"
            f"Контекст:\n{cited_context}\n\nВопрос: {question}"
        )

        llm = OllamaLLM(model=model_name, temperature=temperature)
        prompt = PromptTemplate(
            template="""{input}\n\nОтветь на русском, указывая номера страниц в квадратных скобках.""",
            input_variables=["input"],
        )
        chain = prompt | llm

        send_telegram(chat_id, "🧠 Модель думает…")
        with st.spinner("Генерация ответа…"):
            answer: str = chain.invoke({"input": prompt_text})

        st.success(txt["answer"])
        st.write(answer)
        st.download_button(txt["download"], answer.encode("utf-8"), file_name="answer.txt")
        send_telegram(chat_id, f"✅ Ответ:\n{answer}", inline=True)

        # PostgreSQL логирование
        log_to_db(chat_id, question, answer, model_name, temperature, bad)

        st.caption(f"⌛ Задействована модель: {model_name} | Температура: {temperature}")


if __name__ == "__main__":
    main()