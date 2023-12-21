# Обучение модели hr search 

Этот репозиторий содержит код для обучения модели поиска с использованием модели paraphrase-multilingual-mpnet-base-v2 из библиотеки Hugging Face Transformers.


## Установка

1. Клонируйте репозиторий:
```bash
   git clone https://github.com/AlexWortega/hr_search
   cd hr_search
```
2. Создайте и активируйте виртуальное окружение (опционально, но рекомендуется):
```bash
   python3 -m venv .env
   source .env/bin/activate
```
3. Установите все что вам надо
```bash
   pip install -r requirements.txt
```

3.1. Скачайте данные
Для работы с данными перейдите в директорию data
```bash
cd data
```

3.2. Перейдите в директорию подбора метрик, чтобы ознакомиться с результатами подбора модели и гиперпараметров
```bash
cd metrics
```

4. Запуск трейна
```bash
   python3 train.py --datapath data/train.pkl --modelname paraphrase-multilingual-mpnet-base-v2 --learningrate 1e-5 --batchsize 16 --margin 0.3 --epochs 10 --seed 42 --projectname paraphrase-training --checkpointpath paraphrasecheckpoint.pt --checkpointsteps 1000 --outputpath paraphrasemodel.pt
```


