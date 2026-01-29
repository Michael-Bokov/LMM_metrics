import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from rubert_tiny import EvalClassifier

classifier = EvalClassifier('cointegrated/rubert-tiny2')

# 2. Протестируй на test датасете (zero-shot)
test_df=pd.read_csv("data/rusentiment_test.csv")
test_texts = test_df['text'].tolist()[:100]  # Начни с 100 примеров
true_labels = test_df['label'].tolist()[:100]

# 3. Получи предсказания
classifier = EvalClassifier('cointegrated/rubert-tiny2')
results_tiny = classifier.predict(test_texts)

classifier = EvalClassifier('seara/rubert-base-cased-russian-sentiment')
results_sentiment = classifier.predict(test_texts)
# 4. Вычисли метрики
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Базовые метрики
accuracy = accuracy_score(true_labels, results_tiny['predictions'])
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, results_tiny['predictions'], average='weighted'
)

# Confusion Matrix
cm = confusion_matrix(true_labels, results_tiny['predictions'])

# Время и длина (уже есть в results)
avg_time = sum(results_tiny['inference_times']) / len(results_tiny['inference_times'])
avg_length = sum(results_tiny['text_lengths']) / len(results_tiny['text_lengths'])
print("Model: rubert-tiny2")
print(f"Accuracy: {accuracy:.2%}")
print(f"precision: {precision:.2%}")
print(f"recall: {recall:.2%}")
print(f"f1:{f1:.2%}")
print(f'inference time: {avg_time}')
print(f'average length: {avg_length}')


print("\nConfusion matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(true_labels, results_tiny['predictions']))

print("Model: rubert-base-cased-russian-sentiment")

# Базовые метрики
accuracy = accuracy_score(true_labels, results_sentiment['predictions'])
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, results_sentiment['predictions'], average='weighted'
)

# Confusion Matrix
cm = confusion_matrix(true_labels, results_sentiment['predictions'])

# Время и длина (уже есть в results)
avg_time = sum(results_sentiment['inference_times']) / len(results_sentiment['inference_times'])
avg_length = sum(results_sentiment['text_lengths']) / len(results_sentiment['text_lengths'])

print(f"Accuracy: {accuracy:.2%}")
print(f"precision: {precision:.2%}")
print(f"recall: {recall:.2%}")
print(f"f1:{f1:.2%}")
print(f'inference time: {avg_time}')
print(f'average length: {avg_length}')


print("\nConfusion matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(true_labels, results_sentiment['predictions']))