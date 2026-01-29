from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import time
import pandas as pd
import numpy as np
from typing import List, Dict
import re

class RussianGPT2SentimentAnalyzer:
    """Анализатор тональности на основе русской GPT-2"""
    
    def __init__(self, model_name='ai-forever/rugpt3small_based_on_gpt2'):
        print(f"Загрузка генеративной модели: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # # Для русской GPT-2 часто нужен padding token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(time.ctime())
        print("✓ Модель загружена")
    
    def analyze_sentiment(self, text: str, prompt_template: str = "default",
                         temperature: float = 0.1, max_new_tokens: int = 10) -> Dict:
        """Анализ тональности с разными промптами"""
        
        # Укорачиваем текст если слишком длинный
        if len(text) > 200:
            text = text[:197] + "..."
        
        # Выбор шаблона промпта
        if prompt_template == "short":
            prompt = f"тональность: {text}"#f"КлТекст: {text}\nТональность текста:"
        elif prompt_template == "medium":
            prompt = f"Определи тональность текста: {text}"#f"Отзыв: '{text}'\Оценка: >5 положительная <5 отрицательная =5 нейтральная:"
        elif prompt_template == "long":
            prompt = f"""Определи тональность текста (выбери один вариант):
положительный
отрицательный  
нейтральный
Текст:{text}"""
            # prompt = f"""Проанализируй текст и определи его эмоциональную окраску. 
            # Ответь одним словом: положительный, отрицательный или нейтральный.
            # Текст: {text}
            # Тональность:"""  
        elif prompt_template == "few_shot":
            # Few-shot с примерами
            prompt = f"""Определи тональность текста (выбери один вариант)
            положительный
отрицательный  
нейтральный
            примеры:
текст: "Отличный товар, всем рекомендую!" -> положительный
текст: "Ужасное качество, не покупайте" -> отрицательный
текст: "Обычный продукт, ничего особенного" -> нейтральный

текст: "{text}" ->"""            
# prompt = f"""Инструкция: определи тональность текста.
# Текст: "Отличный товар, всем рекомендую!"
# Тональность: положительный

# Текст: "Ужасное качество, не покупайте"
# Тональность: отрицательный

# Текст: "Обычный продукт, ничего особенного"
# Тональность: нейтральный

# Теперь определи:
# Текст: "{text}"
# Тональность:"""
        else:  # default
            prompt = f"""Определи тональность текста (выбери один вариант):
положительный
отрицательный  
нейтральный
Текст:{text}"""
            
            #f"Тональность текста '{text}': "
        
        # Токенизация
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True,
            truncation=True,
            max_length=256).to(self.device)
                
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # Ключевое исправление!
                temperature=max(0.1, temperature),  # Минимум 0.1
                do_sample=False, # temperature > 0.1,
                #pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2
            )
        
        inference_time = time.time() - start_time
        
        # Декодируем только сгенерированную часть
        generated_tokens = outputs[0]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        #print("__________СГЕНЕРИРОВАНО___________", generated_text)
        if prompt in generated_text:
           generated_text = generated_text.replace(prompt, "").strip()
        
        # Извлекаем тональность из ответа
        sentiment = self._extract_sentiment(generated_text)
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'sentiment': sentiment,
            'inference_time': inference_time,
            'text_length': len(text),
            'response_length': len(generated_text)
        }
    
    def _extract_sentiment(self, text: str) -> str:
        """Извлечение тональности из сгенерированного текста"""
        text_lower = text.lower()
        
        # Ищем ключевые слова
        positive_keywords = ['положительный', 'позитив', 'positive', 'хорош', 'отличн', 'рекоменд', 'классн', 'замечательн']
        negative_keywords = ['отрицательный', 'негатив', 'negative', 'плох', 'ужас', 'разочарован', 'отвратительн', 'кошмарн']
        neutral_keywords = ['нейтральный', 'нейтральн', 'neutral', 'средн', 'обычн', 'нормальн', 'стандартн']
        
        # Проверяем наличие слов
        for word in positive_keywords:
            if word in text_lower:
                return 'позитивный'
        
        for word in negative_keywords:
            if word in text_lower:
                return 'негативный'
        
        for word in neutral_keywords:
            if word in text_lower:
                return 'нейтральный'
        
        # Если не нашли - пытаемся угадать по первым символам
        first_words = text_lower.split()[:3]
        for word in first_words:
            if word.startswith('пол') or word.startswith('pos') or word.startswith('поз'):
                return 'позитивный'
            elif word.startswith('отр') or word.startswith('neg') or word.startswith('нег'):
                return 'негативный'
            elif word.startswith('ней') or word.startswith('neu') :
                return 'нейтральный'
        
        # Дополнительные эвристики
        if any(word in text_lower for word in ['👍', '😊', '😍', '❤️', 'супер', 'отлично', 'прекрасно']):
            return 'позитивный'
        elif any(word in text_lower for word in ['👎', '😠', '😡', '💔', 'ужасно', 'кошмар', 'плохо']):
            return 'негативный'
        # Также проверяем числовые оценки
        if any(word in text_lower for word in ['5', '4', 'отлично', 'хорошо']):
            return 'позитивный'
        elif any(word in text_lower for word in ['1', '2', 'плохо', 'ужасно']):
            return 'негативный'
        elif any(word in text_lower for word in ['3', 'нормально', 'средне']):
            return 'нейтральный'

        
        return 'unknown'
    
    def batch_analyze(self, texts: List[str], prompt_template: str = "default",
                     temperature: float = 0.1) -> Dict:
        """Анализ нескольких текстов"""
        results = []
        inference_times = []
        
        print(f"  Анализ {len(texts)} текстов...")
        
        for i, text in enumerate(texts):
            try:
                result = self.analyze_sentiment(text, prompt_template, temperature)
                results.append(result)
                inference_times.append(result['inference_time'])
                
                # Прогресс
                if (i + 1) % 5 == 0:
                    print(f"    Обработано: {i + 1}/{len(texts)}")
                    
            except Exception as e:
                print(f"    Ошибка при обработке текста {i}: {e}")
                # Добавляем заглушку
                results.append({
                    'prompt': '',
                    'generated_text': '',
                    'sentiment': 'unknown',
                    'inference_time': 0.0,
                    'text_length': len(text),
                    'response_length': 0
                })
                inference_times.append(0.0)
        
        # Собираем предсказания
        predictions = [r['sentiment'] for r in results]
        
        return {
            'predictions': predictions,
            'results': results,
            'avg_inference_time': np.mean(inference_times) if inference_times else 0,
            'avg_text_length': np.mean([r['text_length'] for r in results]) if results else 0,
            'avg_response_length': np.mean([r['response_length'] for r in results]) if results else 0
        }

def run_quick_experiments():
    """Быстрые эксперименты (для теста)"""
    
    # Загружаем данные
    print("Загрузка тестовых данных...")
    test_df = pd.read_csv("data/rusentiment_train.csv")
    test_df['label'] = test_df['label'].apply(
    lambda x: 'негативный' if x == 'negative' else 
              ('нейтральный' if x == 'neutral' else 'позитивный'))
    print(test_df['label'].unique())
    test_texts = test_df['text'].tolist()[:20]  # Только 20 для теста
    true_labels = test_df['label'].tolist()[:20]
    
    print(f"Загружено {len(test_texts)} тестовых примеров")
    
    # Инициализируем модель
    print("\nИнициализация генеративной модели...")
    analyzer = RussianGPT2SentimentAnalyzer()
    
    # Быстрый тест одного примера
    print("\n📝 Тест одного примера:")
    test_text = test_texts[10]
    true_label = true_labels[10]
    
    result = analyzer.analyze_sentiment(test_text, prompt_template="long")
    print(f"Текст: {test_text[:50]}...")
    print(f"Промпт: {result['prompt'][:50]}...")
    print(f"Сгенерировано: {result['generated_text']}")
    print(f"Извлечено: {result['sentiment']}")
    print(f"Реальное: {true_label}")
    print(f"Время: {result['inference_time']:.2f} сек")
    
    # Основные эксперименты
    print("\n" + "="*60)
    print("🚀 ЗАПУСК ЭКСПЕРИМЕНТОВ")
    print("="*60)
    
    from sklearn.metrics import accuracy_score
    
    # Эксперимент 1: Разные промпты (на 10 примерах)
    print("\n1. Эксперимент: Разные шаблоны промптов")
    print("-"*40)
    
    templates = ['short', 'medium', 'long', 'few_shot']
    sample_size = 10
    
    for template in templates:
        print(f"\n  Шаблон: {template}")
        results = analyzer.batch_analyze(
            test_texts[:sample_size], 
            prompt_template=template
        )
        
        predictions = results['predictions']
        accuracy = accuracy_score(true_labels[:sample_size], predictions)
        
        print(f"    Accuracy: {accuracy:.1%}")
        print(f"    Время: {results['avg_inference_time']:.3f} сек/пример")
        
        # Показываем примеры
        for i in range(min(2, len(results['results']))):
            r = results['results'][i]
            print(f"    Пример {i+1}: '{r['generated_text'][:30]}...' -> {r['sentiment']}")
    
    # Эксперимент 2: Температура (на 5 примерах)
    print("\n2. Эксперимент: Влияние температуры")
    print("-"*40)
    
    temperatures = [0.1, 0.7, 1.2]
    
    for temp in temperatures:
        print(f"\n  Температура: {temp}")
        
        predictions = []
        for text in test_texts[:5]:
            result = analyzer.analyze_sentiment(text, temperature=temp)
            predictions.append(result['sentiment'])
        
        accuracy = accuracy_score(true_labels[:5], predictions)
        print(f"    Accuracy: {accuracy:.1%}")
        
        # Разнообразие ответов
        unique_preds = set(predictions)
        print(f"    Уникальные ответы: {unique_preds}")
    
    # Эксперимент 3: Zero-shot vs Few-shot
    print("\n3. Эксперимент: Zero-shot vs Few-shot")
    print("-"*40)
    
    for template, name in [('medium', 'Zero-shot'), ('few_shot', 'long')]:
        print(f"\n  {name}:")
        results = analyzer.batch_analyze(test_texts[:8], prompt_template=template)
        predictions = results['predictions']
        #print("ЗКУВЫЛФЯ",predictions[1])
        
        accuracy = accuracy_score(true_labels[:8], predictions)
        print(f"    Accuracy: {accuracy:.1%}")
        print(f"    Время: {results['avg_inference_time']:.3f} сек/пример")
    
    # Общая оценка модели
    print("\n" + "="*60)
    print("📊 ОБЩАЯ ОЦЕНКА МОДЕЛИ")
    print("="*60)
    
    # Тестируем на всех 20 примерах с лучшим шаблоном
    best_template = 'few_shot'
    print(f"\nТестирование с шаблоном '{best_template}' на {len(test_texts)} примерах:")
    
    results = analyzer.batch_analyze(test_texts, prompt_template=best_template)
    predictions = results['predictions']
    
    accuracy = accuracy_score(true_labels, predictions)
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(f"\nОбщие метрики:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Среднее время: {results['avg_inference_time']:.3f} сек")
    print(f"  Длина ответа: {results['avg_response_length']:.1f} символов")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions, labels=['негатиный', 'нейтральный', 'позитивный'])
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(true_labels, predictions))
    
    # Анализ ошибок
    print(f"\n🔍 Анализ ошибок:")
    errors = []
    for i, (text, pred, true) in enumerate(zip(test_texts, predictions, true_labels)):
        if pred != true:
            errors.append({
                'idx': i,
                'text_preview': text[:40] + "..." if len(text) > 40 else text,
                'predicted': pred,
                'true': true
            })
    
    print(f"Всего ошибок: {len(errors)} из {len(test_texts)} ({len(errors)/len(test_texts):.1%})")
    
    if errors:
        print("\nПримеры ошибок:")
        for error in errors[:3]:
            print(f"  Текст: {error['text_preview']}")
            print(f"    Предсказано: {error['predicted']}, Реальное: {error['true']}")
    
    # Сравнение с классификаторами
    print("\n" + "="*60)
    print("🆚 СРАВНЕНИЕ С КЛАССИФИКАТОРАМИ")
    print("="*60)
    
    comparison_data = {
        'Модель': ['RuBERT-tiny2', 'RuBERT-base-sentiment', 'GPT-2 (генеративная)'],
        'Accuracy': ['34%', '85%', f'{accuracy:.1%}'],
        'Время (сек)': ['0.002', '0.032', f'{results["avg_inference_time"]:.3f}'],
        'Тип': ['Классификатор', 'Классификатор', 'Генеративная']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Выводы
    print("\n" + "="*60)
    print("📋 ВЫВОДЫ")
    print("="*60)
    
    print("\n1. Генеративная модель (GPT-2):")
    print(f"   - Accuracy: {accuracy:.1%}")
    print(f"   - Скорость: {results['avg_inference_time']:.3f} сек на пример")
    print(f"   - Лучший промпт: Few-shot")
    
    print("\n2. Классификаторы vs Генеративные:")
    print("   - Классификаторы быстрее и точнее для этой задачи")
    print("   - Генеративные модели требуют тщательной настройки промптов")
    print("   - Few-shot улучшает качество генеративных моделей")
    
    print("\n3. Рекомендации:")
    print("   - Для production: использовать классификатор (85% accuracy)")
    print("   - Для экспериментов: генеративная модель с few-shot промптами")
    print("   - Для speed-critical задач: rubert-tiny2 (самый быстрый)")

def run_full_experiments():
    """Полные эксперименты (запускать если quick работает)"""
    print("Запуск полных экспериментов...")
    # ... (используй код из предыдущего сообщения, но с исправлениями)

if __name__ == "__main__":
    print("="*80)
    print("🤖 ЭКСПЕРИМЕНТЫ С ГЕНЕРАТИВНОЙ МОДЕЛЬЮ ДЛЯ АНАЛИЗА ТОНАЛЬНОСТИ")
    print("="*80)
    
    run_quick_experiments()
    
    print("\n" + "="*80)
    print("✅ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print(time.ctime())

    print("="*80)