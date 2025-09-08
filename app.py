import os
import openai
import httpx
import logging
import json
import random
import datetime
import tiktoken
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage


def count_tokens(text: str, model: str) -> int:
    """Подсчитывает количество токенов в тексте"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Если модель не найдена, используем cl100k_base (для GPT-3.5/4)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def log_json(event_type: str, content: str, session_id: str, tokens: int = 0):
    """Логирует событие в JSON формате"""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": event_type,
        "content": content,
        "session_id": session_id,
        "tokens": tokens
    }
    logging.info(json.dumps(log_entry, ensure_ascii=False))


def setup_openai():
    """Настройка доступа к OpenAI API"""
    # Загружаем переменные окружения из .env файла
    load_dotenv()

    # Проверяем наличие API ключа
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY не найден в переменных окружения. Создайте .env файл с вашим API ключом.")

    print("OpenAI API настроен успешно")


def chat_loop(conversation: ConversationChain, system_message, session_id: str, model_name: str):
    """Основной цикл диалога с пользователем"""
    input_tokens = 0   # Счетчик входящих токенов (от пользователя)
    output_tokens = 0  # Счетчик исходящих токенов (от бота)
    total_tokens = 0   # Счетчик общего количества токенов

    while True:
        try:
            user_text = input("Вы: ")
            user_tokens = count_tokens(user_text, model_name)
            input_tokens += user_tokens
            total_tokens += user_tokens
            log_json("user_message", user_text, session_id, user_tokens)
        except (KeyboardInterrupt, EOFError):
            print("\nБот: Завершение работы.")
            log_json(
                "session_end", f"Session ended by user. Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}", session_id)
            break

        user_text = user_text.strip()
        if user_text == "":
            continue

        command = user_text.lower()
        if command in ("выход", "стоп", "конец", "quit", "exit"):
            print("Бот: До свидания!")
            log_json(
                "session_end", f"Session ended by command. Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}", session_id)
            break
        if command in ("сброс",):
            conversation.memory.clear()
            conversation.memory.chat_memory.add_message(system_message)
            print("Бот: Контекст диалога очищен.")
            log_json(
                "context_reset", f"Context cleared. Current stats - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}", session_id)
            continue

        # Обработка команды /order
        if command.startswith("/order "):
            order_id = command.split(" ", 1)[1]
            try:
                with open('data/orders.json', 'r', encoding='utf-8') as f:
                    orders = json.load(f)
                
                if order_id in orders:
                    order_info = orders[order_id]
                    # Модифицируем запрос пользователя, добавляя информацию о заказе
                    enhanced_query = f"Пользователь спрашивает о заказе {order_id}. Информация о заказе: {json.dumps(order_info, ensure_ascii=False)}. Ответь на русском языке."
                else:
                    enhanced_query = f"Пользователь спрашивает о заказе {order_id}, но такого заказа не существует в системе."
                
                # Отправляем модифицированный запрос в LLM
                bot_reply = conversation.predict(input=enhanced_query).strip()
                
            except Exception as e:
                enhanced_query = f"Ошибка при получении информации о заказе {order_id}: {e}"
                bot_reply = conversation.predict(input=enhanced_query).strip()
            
            print(f"Бот: {bot_reply}")
            bot_tokens = count_tokens(bot_reply, model_name)
            output_tokens += bot_tokens
            total_tokens += bot_tokens
            log_json("bot_reply", bot_reply, session_id, bot_tokens)
            continue

        try:
            # Получаем ответ от модели и сразу очищаем от лишних пробелов в начале и конце строки при помощи strip()
            bot_reply = conversation.predict(input=user_text).strip()

        except openai.APITimeoutError as e:
            print("Бот: [Ошибка] Превышено время ожидания ответа.")
            log_json("error", f"APITimeoutError: {e}", session_id)
            continue
        except openai.APIConnectionError as e:
            print("Бот: [Ошибка] Не удалось подключиться к сервису LLM.")
            log_json("error", f"APIConnectionError: {e}", session_id)
            continue
        except openai.AuthenticationError as e:
            print("Бот: [Ошибка] Проблема с API-ключом (неавторизовано).")
            log_json(
                "error", f"AuthenticationError: {e}. Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}", session_id)
            break  # здесь можно завершить, т.к. дальнейшая работа бессмысленна
        except openai.PermissionDeniedError as e:
            print("Бот: [Ошибка] Доступ запрещен. Проверьте настройки API.")
            log_json(
                "error", f"PermissionDeniedError: {e}. Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}", session_id)
            break
        except httpx.ReadTimeout as e:
            print("Бот: [Ошибка] Превышено время ожидания чтения ответа.")
            log_json("error", f"ReadTimeout: {e}", session_id)
            continue
        except httpx.ConnectTimeout as e:
            print("Бот: [Ошибка] Превышено время ожидания подключения.")
            log_json("error", f"ConnectTimeout: {e}", session_id)
            continue
        except openai.OpenAIError as e:
            print(f"Бот: [Ошибка API] {e}")
            log_json("error", f"OpenAIError: {e}", session_id)
            continue
        except Exception as e:
            print(f"Бот: [Неизвестная ошибка] {e}")
            log_json("error", f"UnknownError: {e}", session_id)
            continue
        print(f"Бот: {bot_reply}")
        bot_tokens = count_tokens(bot_reply, model_name)
        output_tokens += bot_tokens
        total_tokens += bot_tokens
        log_json("bot_reply", bot_reply, session_id, bot_tokens)

    # Логируем финальную статистику при завершении цикла
    log_json("session_final_stats",
             f"Final session statistics - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}", session_id)


def main():
    # Генерируем уникальное имя файла для сессии
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/session_{session_id}.jsonl"

    # Создаем папку logs если её нет
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        # Для JSON Lines нужен только message без timestamp
        format="%(message)s",
        filemode='w'
    )

    # Настройка доступа: загрузка переменных окружения, установка ключа и адреса API
    setup_openai()

    # Создаём модель и память
    model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    chat_model = ChatOpenAI(
        model_name=model_name,
        openai_api_base=os.getenv(
            "OPENAI_API_BASE", "https://api.essayai.ru/v1"),
        request_timeout=30
    )
    memory = ConversationBufferMemory()

    with open('data/faq.json', 'r', encoding='utf-8') as f:
        faq_data = json.load(f)

    faq_json = json.dumps(faq_data, ensure_ascii=False, indent=2)

    # Системное сообщение — общие правила для ассистента
    system_message = SystemMessage(
        content=f"""Ты — вежливый и точный ассистент поддержки магазина «Shoply». Будь максимально кратким. Отвечай только на русском языке.

        ПРАВИЛА ОТВЕТОВ:
        1. Если пользователь спрашивает о конкретном заказе и тебе предоставлена информация о заказе - отвечай на основе этой информации. НЕ вызывай менеджера.

        2. Для общих вопросов о заказах, доставке, оплате, возвратах, промокодах - отвечай ТОЛЬКО на основе FAQ ниже. Если точного ответа нет в FAQ, скажи: 'Приглашаю менеджера. Он ответит вам через {random.randint(1, 20)} минут.'

        3. Для общих вопросов (приветствие, благодарность, общие вопросы о магазине, погоде и т.д.) - отвечай самостоятельно, не вызывая менеджера.

        4. Если вопрос касается конкретного заказа, но информации о нем нет - так и скажи, что информации о нем нет.
        5. На вопросы, которые не относятся к работе магазина - отвечай, что ты не можешь ответить на этот вопрос.

        FAQ: {faq_json}""")

    # Цепочка диалога на основе модели и памяти
    conversation = ConversationChain(llm=chat_model, memory=memory)

    memory.chat_memory.add_message(system_message)

    log_json("session_start", "=== New session ===", session_id)
    print("Привет! Я консольный бот. Для выхода напишите «выход».")
    chat_loop(conversation, system_message, session_id, model_name)


if __name__ == "__main__":
    main()
