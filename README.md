# fake-classification-kontur

# FakeClassifier - тестовое задание [Контур.](https://kontur.ru)

## To-do list

- [x] Ресерч
- [x] Поиск подходящей модели для дообучения SkolkovoInstitute/russian_toxicity_classifier 
- [x] Эксперименты с лемматизацией и стопсловами
- [x] Подбор параметров
- [x] Создание скриптов для обучения
- [x] Создание скриптов для предсказаний
- [ ] Создание мини-сервиса на flask
- [ ] Docker

## Ресерч (микровыжимка)

**WordCloud**
![alt text](img/wordcloud.png)

Я попробовал достаточно много классических моделей

1. TF-IDF + SGD:
![alt text](img/tfidf_sgd.png)
2. TF-IDF + MultinomialNB:
![alt text](img/tfidf_MultinomialNB.png)
3. LightAutoML with DeepPavlov Bert:
![alt text](img/lama.png)

Также я эксперементировал со стоп-словами и лемматизацией - ничего хорошего из этого не вышло, но в коде возможность удалять стоп-слова и получать лемматизированные тексты я оставил.

В итоге мой наилучший скор на тестовых данных:
![alt text](img/best.png)


**Инструкция:**

- настройте под себя параметры в `config.yml`, если решаете свою задачу.
- если необходимо, вы можете изменить предобработку данных внутри `solution/data.py`
- модель находится в `solution/model.py`
- Обучение модели `python solution/train.py`
- Запуск из консоли `python solution/predict.py -f [FILE PATH]`
- Или `python solution/predict.py -t [TEXT]`
