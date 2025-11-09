import streamlit as st
import joblib
import torch
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

st.set_page_config(page_title="NLP Project", layout="wide")

# В начале файла, где выбирается страница, добавь третью страницу:
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите страницу:",
    [
        # "Страница 1 • Классификация отзывов на рестораны",
        "Страница 1 • Классификация новостей из Telegram", 
        "Страница 2 • Генерация текста в стиле Форреста Гампа"
    ]
)

# =========================
# СТРАНИЦА 1
# =========================
# if page.startswith("Страница 1"):
#     st.title("Классификация отзывов на рестораны")

# =========================
# СТРАНИЦА 2
# =========================
if page.startswith("Страница 1"):
    st.title("Классификация новостей из Telegram-каналов")
    st.divider()

    # Поле ввода
    user_input = st.text_area("Введите текст новости:", height=150)
    if st.button("Классифицировать"):
        if not user_input.strip():
            st.warning("Пожалуйста, введите текст новости.")
            st.stop()

        # -------------------------
        # Загружаем TF-IDF + LogisticRegression
        # -------------------------
        start_time = time.time()
        tfidf = joblib.load("/home/ninakhay/ds_bootcamp/ds-phase-2/nlp_project/models/tfidf_vectorizer.pkl")
        lr_model = joblib.load("/home/ninakhay/ds_bootcamp/ds-phase-2/nlp_project/models/logistic_regression_model.pkl")

        user_tfidf = tfidf.transform([user_input])
        tfidf_pred = lr_model.predict(user_tfidf)[0]
        tfidf_time = time.time() - start_time

        # -------------------------
        # LSTM
        # -------------------------
        vocab = joblib.load("/home/ninakhay/ds_bootcamp/ds-phase-2/nlp_project/models/vocab.pkl")
        label_encoder = joblib.load("/home/ninakhay/ds_bootcamp/ds-phase-2/nlp_project/models/label_encoder.pkl")

        # === Определяем класс модели ===
        class LSTMClassifier(torch.nn.Module):
            def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, output_dim=5):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
                self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.fc = torch.nn.Linear(hidden_dim*2, output_dim)

            def forward(self, x):
                x = self.embedding(x)
                _, (hidden, _) = self.lstm(x)
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
                return self.fc(hidden)

        # === Создаем и загружаем веса ===
        model_lstm = LSTMClassifier(vocab_size=len(vocab), output_dim=len(label_encoder.classes_))
        state_dict = torch.load("/home/ninakhay/ds_bootcamp/ds-phase-2/nlp_project/models/lstm_weights.pt", map_location="cpu")
        model_lstm.load_state_dict(state_dict)
        model_lstm.eval()

        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r"[^а-яё\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        def text_to_tensor(text, max_len=50):
            tokens = preprocess_text(text).split()
            indices = [vocab.get(t, vocab["<unk>"]) for t in tokens]
            if len(indices) < max_len:
                indices += [vocab["<pad>"]] * (max_len - len(indices))
            else:
                indices = indices[:max_len]
            return torch.tensor([indices], dtype=torch.long)

        tensor = text_to_tensor(user_input)
        with torch.no_grad():
            out = model_lstm(tensor)
        pred_id = out.argmax(dim=1).item()
        lstm_pred = label_encoder.inverse_transform([pred_id])[0]
        lstm_time = time.time() - start_time

        # -------------------------
        # RuBERT tiny2
        # -------------------------
        start_time = time.time()
        model_name = "/home/ninakhay/ds_bootcamp/ds-phase-2/nlp_project/models/rubert_tiny_model"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_rubert = AutoModelForSequenceClassification.from_pretrained(model_name)
        model_rubert.eval()

        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model_rubert(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        rubert_pred = label_encoder.inverse_transform([pred_id])[0]
        rubert_time = time.time() - start_time

        # -------------------------
        # Вывод результатов
        # -------------------------
        st.success("Результаты классификации:")
        col1, col2, col3 = st.columns(3)
        col1.metric("TF-IDF + LogisticRegression", tfidf_pred, f"{tfidf_time:.2f} сек")
        col2.metric("LSTM", lstm_pred, f"{lstm_time:.2f} сек")
        col3.metric("RuBERT-tiny2", rubert_pred, f"{rubert_time:.2f} сек")

        st.divider()

        # -------------------------
        # График F1-macro
        # -------------------------
        st.subheader("Сравнение F1-macro моделей")

        f1_scores = {
            "TF-IDF + LR": 0.84,
            "LSTM": 0.62,  
            "RuBERT-tiny2": 0.94
        }

        fig, ax = plt.subplots(figsize=(3,2))
        ax.bar(f1_scores.keys(), f1_scores.values())
        ax.set_title("F1-macro по моделям")
        ax.set_ylim(0,1)
        for i, v in enumerate(f1_scores.values()):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

        # -------------------------
        # МЕТРИКИ BERT
        # -------------------------
        st.subheader("Метрики RuBERT-tiny2 на валидации")
        bert_cols = st.columns(4)
        bert_cols[0].metric("Accuracy", "0.9527")
        bert_cols[1].metric("Precision", "0.9538")
        bert_cols[2].metric("Recall", "0.9540")
        bert_cols[3].metric("F1-score", "0.9536")

        st.divider()

        # -------------------------
        # Матрицы ошибок
        # -------------------------
        st.subheader("Матрицы ошибок по моделям")

        st.write("Матрица ошибок TF-IDF:")
        st.image("/home/ninakhay/ds_bootcamp/ds-phase-2/nlp_project/images/confmat_tfidf.png")

        st.write("Матрица ошибок LSTM:")
        st.image("/home/ninakhay/ds_bootcamp/ds-phase-2/nlp_project/images/confmat_lstm.png")

        st.write("Матрица ошибок RuBERT:")
        st.image("/home/ninakhay/ds_bootcamp/ds-phase-2/nlp_project/images/confmat_rubert.png")


# =========================
# СТРАНИЦА 3
# =========================
elif page.startswith("Страница 2"):
    st.title("Генерация текста в стиле Лесного болвана")
    st.divider()
    
    @st.cache_resource
    def load_gpt_model():
        try:
            # Загружаем токенизатор
            tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
            
            # ПРЯМОЙ ПУТЬ к папке с моделью
            model_path = '/home/ninakhay/ds_bootcamp/ds-phase-2/nlp_project/models/model_gump'
            # Проверяем существует ли папка
            if not os.path.exists(model_path):
                st.error(f"Папка '{model_path}' не найдена!")
                # Показываем что есть в текущей директории
                st.write("Файлы в текущей директории:")
                for item in os.listdir('.'):
                    st.write(f"- {item}")
                return None, None
            
            # Загружаем модель
            model = GPT2LMHeadModel.from_pretrained(model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model.eval()
            # st.success("Модель загружена успешно!")
            return tokenizer, model
            
        except Exception as e:
            st.error(f"Ошибка загрузки: {str(e)}")
            return None, None
    
    tokenizer, model = load_gpt_model()
    
    if tokenizer is None or model is None:
        st.stop()
    
    # Настройки генерации
    st.subheader("Настройки генерации")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_length = st.slider("Длина текста", 50, 300, 100)
    
    with col2:
        temperature = st.slider("Температура", 0.1, 1.5, 0.9, 0.1)
    
    with col3:
        num_sequences = st.slider("Число вариантов", 1, 3, 1)
    
    # Промпт
    st.subheader("Введите начало истории")
    prompt = st.text_area(
        "Начните свою историю:",
        height=100
    )
    
    # Генерация
    if st.button("Сгенерировать историю", type="primary"):
        if not prompt.strip():
            st.warning("Введите начало истории.")
        else:
            with st.spinner("Лесной думает..."):
                try:
                    # Токенизация
                    inputs = tokenizer.encode(prompt, return_tensors='pt')
                    
                    # Генерация
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            max_length=max_length,
                            num_return_sequences=num_sequences,
                            temperature=temperature,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.1
                        )
                    
                    # Вывод результатов
                    st.success("Результат:")
                    
                    for i, output in enumerate(outputs):
                        text = tokenizer.decode(output, skip_special_tokens=True)
                        st.write(f"**Вариант {i+1}:**")
                        st.write(text)
                        # st.code(text)
                        
                        if i < len(outputs) - 1:
                            st.divider()
                            
                except Exception as e:
                    st.error(f"Ошибка генерации: {str(e)}")