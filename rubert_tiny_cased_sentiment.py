from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
import time
class EvalClassifier:
    def __init__(self,model_name):
        self.model_name = model_name #seara/rubert-base-cased-russian-sentiment "cointegrated/rubert-tiny2"
        print(f"Загрузка модели: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3  # positive, negative, neutral
        )
        # Определяем метки
        if model_name == 'cointegrated/rubert-tiny2': #hasattr(self.model.config, 'id2label'):
            self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        else:
            self.label_map = self.model.config.id2label
        
        # self.label_map = {
        #     0: "negative",
        #     1: "neutral", 
        #     2: "positive"
        # }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, texts: List[str], batch_size: int = 16) -> Dict:
        """Предсказание тональности для списка текстов"""
        predictions = []
        probabilities = []
        inference_times = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            start_time = time.time()
            
            # Токенизация
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                batch_preds = torch.argmax(probs, dim=-1)
            
            inference_time = time.time() - start_time
            
            # Сохраняем результаты
            for pred, prob in zip(batch_preds.cpu().numpy(), probs.cpu().numpy()):
                predictions.append(self.label_map[pred])
                probabilities.append(prob.tolist())
                inference_times.append(inference_time / len(batch))
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "inference_times": inference_times,
            "text_lengths": [len(text) for text in texts]
        }
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Получение эмбеддингов для semantic similarity"""
        embeddings = []
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.bert(**inputs)
            # Используем эмбеддинг [CLS] токена
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = cls_embeddings.cpu().numpy().tolist()
        
        return embeddings
