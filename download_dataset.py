from datasets import load_dataset
import pandas as pd

def load_rusentiment():
    """Загрузка датасета RuSentiment"""
    print("Загрузка RuSentiment датасета...")
    
    # Загружаем с Hugging Face
    #dataset = load_dataset("blanchefort/rusentiment")
    
    # Преобразуем в pandas DataFrame
    #train_df = pd.DataFrame(dataset['train'])
    #test_df = pd.DataFrame(dataset['test'])
    try:
        df = pd.read_csv("rusentiment_preselected_posts.csv")
        df = df[~df['label'].isin(['speech', 'skip'])]
        print(f"✓ RuSentiment загружен локально")
        
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        #return train_df, test_df
    except:
        pass
    
    print(f"Размер тренировочных данных: {len(train_df)}")
    print(f"Размер тестовых данных: {len(test_df)}")
    print(f"Классы: {train_df['label'].unique()}")
    print(f"Распределение классов:")
    print(train_df['label'].value_counts())
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_rusentiment()
    # Сохраняем для дальнейшего использования
    train_df.to_csv("data/rusentiment_train.csv", index=False)
    test_df.to_csv("data/rusentiment_test.csv", index=False)