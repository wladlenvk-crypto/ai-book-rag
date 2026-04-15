import os
from openai import OpenAI
from supabase import create_client

# Ключи из Secrets
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def clear_openai_table():
    """Очищает ТОЛЬКО таблицу для OpenAI"""
    print("Очистка таблицы documents_openai...")
    # Удаляем все записи, где id больше 0 (безопасный способ очистки)
    supabase.table("documents_openai").delete().neq("id", 0).execute()
    print("Таблица очищена.")

def split_text(text, chunk_size=2000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def upload_in_batches(file_path, batch_size=5):
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден!")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Делим весь текст на части
    all_chunks = split_text(full_text)
    
    # --- ТЕСТОВЫЙ РЕЖИМ: Берем только первые 2 куска ---
    chunks = all_chunks[:2] 
    print(f"ТЕСТ: Загружаем только {len(chunks)} фрагмента из {len(all_chunks)}.")
    # --------------------------------------------------

    clear_openai_table()

    # Загружаем пачками (batch) для экономии времени и стабильности
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            # 1. Получаем эмбеддинги для всей пачки сразу
            resp = client.embeddings.create(input=batch, model="text-embedding-3-small")
            embeddings = [item.embedding for item in resp.data]

            # 2. Формируем данные для Supabase
            rows = []
            for j, chunk_text in enumerate(batch):
                rows.append({
                    "content": chunk_text,
                    "metadata": {"source": file_path, "part": i + j + 1},
                    "embedding": embeddings[j]
                })

            # 3. Массовая вставка в Supabase
            supabase.table("documents_openai").insert(rows).execute()
            print(f"Загружена пачка: {i + 1} - {i + len(batch)}")
            
        except Exception as e:
            print(f"Ошибка на пачке: {e}")

if __name__ == "__main__":
    if OPENAI_API_KEY and SUPABASE_URL:
        upload_in_batches("book.txt")
    else:
        print("Ошибка: Проверьте Secrets в Hugging Face!")
