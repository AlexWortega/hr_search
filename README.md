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
## Обучение
| Model Name                          | metric # | Epochs | Learning Rate (lr) | Loss Function         |
|-------------------------------------|--------------|--------|--------------------|-----------------------|
| paraphrase-multilingual-mpnet-base-v2 | 0.54           | 3      | 5e-5              |  Contrastive Loss     |
| paraphrase-multilingual-mpnet-base-v2 | 0.43           | 4      | 3e-5              | Contrastive Loss      |
| paraphrase-multilingual-mpnet-base-v2 | 0.75           | 5      | 2e-5              | Triplet Loss          |
| ruElectra                           | 0.56           | 3      | 4e-5              | Cross-Entropy Loss    |
| ruElectra                           | 0.61            | 5      | 3e-5              |  Contrastive Loss            |
| ruElectra                           | 0.68              | 4      | 5e-5              | Triplet Loss  |
| SBERT                               | 0.61            | 3      | 2e-5              |  Contrastive Loss   |
| SBERT                               | 0.594           | 4      | 3e-5              |  Contrastive Loss |
| SBERT                               | 0.64           | 5      | 1e-5              | Triplet Loss  |
## Итоговая метрика нашего обученого подхода в сравнении с открытыми решениями
| Model Name          | MRR Score                                  |
|---------------------|---------------------------------------------|
| paraphrase-multilingual-mpnet-base-v2                | 0.75|
| ruElectra             | 0.68                  |
| SBERT               |0.64 |

