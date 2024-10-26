import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split  # для разделения данных
from sklearn.feature_extraction.text import TfidfVectorizer  # для преобразования текста в числа
from sklearn.linear_model import PassiveAggressiveClassifier  # для классификации
from sklearn.metrics import accuracy_score, confusion_matrix  # для оценки модели




#Функция для очистки текста от ненужных символов
def clean_text(text):

    text = re.sub('r[^a-zA-z\s]', '', text) #Оставляем только буквы верхнего и нижнего регистра и пробелы
    text = text.lower() #Приводим весь текст к нижнему регистру
    text = text.strip() #Убираем лишние пробелы
    return text


if __name__ == "__main__":
    data = pd.read_csv("fake_news.csv")
    data['text'] = data['text'].apply(clean_text)

    X = data['text'] # Признаки — это колонка с текстом
    Y = data['label'] # Метки — это колонка с метками (Fake/True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #Распределяем данные на тренировочные и тестовые в отношении 80 на 20


    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) # Создаем объект TfidfVectorizer с параметрами, игнорируем стандартный список стоп-слов на англ. языке, так же игнорируем слова, которые встречаются более чем в 70% заголовках
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train) # Преобразуем текст в числовые значения
    X_test_tfidf = tfidf_vectorizer.transform(X_test) # Преобразуем текст в числовые значения

    # Создаём объект модели, количество итераций 50
    pac = PassiveAggressiveClassifier(max_iter=50)
    # Обучаем модель на тренировочных данных
    pac.fit(X_train_tfidf, Y_train)
    # Предсказания на тестовых данных
    y_pred = pac.predict(X_test_tfidf)
    # Опеределяем точность модели
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Точность модели: {accuracy * 100:.2f}%")



