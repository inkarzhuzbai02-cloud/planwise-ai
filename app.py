import io, os, json
import cv2, fitz
import numpy as np
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

MODEL = "gemini-flash-lite-latest"

PROMPT = """Ты опытный архитектор-консультант. Проанализируй план здания
на изображении с учётом требований пользователя. Ответь ТОЛЬКО JSON:
{
 "summary": "вывод 2-3 предложения",
 "rooms": [{"name": "", "area_m2": "", "zone": ""}],
 "connections": ["Кухня - Гостиная: прямая связь"],
 "problems": [{"severity": "высокая/средняя/низкая", "issue": "", "advice": ""}],
 "recommendations": [""]
}
Если что-то на плане не читается, так и напиши. Ничего не выдумывай."""

def get_key():
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

def load_image(f):
    data = f.read()
    if f.type == "application/pdf":
        doc = fitz.open(stream=data, filetype="pdf")
        data = doc[0].get_pixmap(dpi=150).tobytes("png")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1600, 1600))
    return img

def enhance(img):
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return Image.fromarray(g).convert("RGB")

def to_md(r):
    md = "# Отчёт PLANWISE AI\n\n" + r.get("summary", "") + "\n\n## Помещения\n"
    for x in r.get("rooms", []):
        md += f"- {x.get('name')} ({x.get('area_m2')} м²), {x.get('zone')}\n"
    md += "\n## Связи\n" + "\n".join("- " + c for c in r.get("connections", []))
    md += "\n\n## Проблемы\n"
    for p in r.get("problems", []):
        md += f"- [{p.get('severity')}] {p.get('issue')}. Совет: {p.get('advice')}\n"
    md += "\n## Рекомендации\n" + "\n".join("- " + c for c in r.get("recommendations", []))
    return md

st.set_page_config(page_title="PLANWISE AI", page_icon="🏠")
st.title("🏠 PLANWISE AI")
st.caption("Анализ архитектурных планов с помощью ИИ")

lang = st.selectbox("Язык отчёта", ["Русский", "Қазақша"])
file = st.file_uploader("Загрузите план", type=["jpg", "jpeg", "png", "pdf"])
req = st.text_area("Требования", placeholder="Например: 3-комнатная квартира, кухня-гостиная, санузел с окном")
use_cv = st.checkbox("Улучшить контраст (OpenCV)", value=True)

if st.button("Анализировать", type="primary"):
    key = get_key()
    if not key:
        st.error("Не найден GEMINI_API_KEY")
        st.stop()
    if not file:
        st.warning("Сначала загрузите план")
        st.stop()
    img = load_image(file)
    st.image(img, caption="Загруженный план")
    with st.spinner("ИИ анализирует план..."):
        try:
            client = genai.Client(api_key=key, http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=5)))
            text = PROMPT + "\nЯзык ответа: " + lang + "\nТребования: " + (req or "не указаны")
            resp = client.models.generate_content(
                model=MODEL,
                contents=[enhance(img) if use_cv else img, text],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            r = json.loads(resp.text)
        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.stop()

    st.subheader("Вывод")
    st.write(r.get("summary", ""))
    st.subheader("Помещения")
    st.table(r.get("rooms", []))
    st.subheader("Связи")
    for c in r.get("connections", []):
        st.write("•", c)
    st.subheader("Возможные проблемы")
    for p in r.get("problems", []):
        st.warning(f"**{p.get('severity')}**: {p.get('issue')}\n\nСовет: {p.get('advice')}")
    st.subheader("Рекомендации")
    for c in r.get("recommendations", []):
        st.success(c)
    st.download_button("Скачать отчёт", to_md(r), file_name="report.md")
    st.caption("Это рекомендации для проверки, а не окончательный вердикт.")
