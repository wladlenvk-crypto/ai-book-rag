import os
from openai import OpenAI
from supabase import create_client

# Берем ключи из переменных окружения (секретов)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_text_to_openai(text_parts):
    print(f"Начинаю загрузку {len(text_parts)} фрагментов...")
    for i, chunk in enumerate(text_parts):
        try:
            # Создаем эмбеддинг
            response = client.embeddings.create(
                input=chunk,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding

            # Записываем в таблицу
            data = {
                "content": chunk,
                "metadata": {"source": "manual_upload", "index": i},
                "embedding": embedding
            }
            supabase.table("documents_openai").insert(data).execute()
            print(f"Успешно загружен фрагмент №{i+1}")
        except Exception as e:
            print(f"Ошибка на фрагменте {i}: {e}")

if __name__ == "__main__":
    # Вставьте сюда текст ваших двух частей книги
    my_book_chunks = [
        "Текст первой части книги...",
        "Текст второй части книги..."
    ]
    
    if OPENAI_API_KEY and SUPABASE_URL:
        upload_text_to_openai(my_book_chunks)
    else:
        print("Ошибка: Ключи не найдены в Secrets!")
