import ollama
from ddgs import DDGS
from datetime import datetime


def create_system_message() -> dict:
    return {
        "role": "system",
        "content": (
            f"You are a precise and factual AI assistant.\n"
            f"Today's date is {datetime.now().strftime('%Y-%m-%d')}.\n\n"
            "Rules:\n"
            "- Answer only the question asked, nothing extra.\n"
            "- Keep answers under 2-3 sentences.\n"
            "- If unsure, respond with 'I don't know'.\n"
        )
    }


def needs_web_search(query: str) -> bool:
    """Check if query requires fresh web information."""
    keywords = ["latest", "news", "today", "current", "update", "2025", "2026"]
    query = query.lower()
    return any(k in query for k in keywords)


def search_web(query: str) -> str:
    """Fetch latest information from DuckDuckGo."""
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"{r['title']} - {r['body']}")
    except Exception as e:
        return f"Search error: {e}"

    return "\n".join(results) if results else "No relevant results found."


def chat(conversation_history: list, user_input: str) -> str:
    """Send conversation + optional web info to LLaMA."""

    if needs_web_search(user_input):
        web_results = search_web(user_input)

        if "No relevant results found" not in web_results:
            content = f"{user_input}\n\nWeb info:\n{web_results}"
        else:
            content = user_input
    else:
        content = user_input

    conversation_history.append({
        "role": "user",
        "content": content
    })

    trimmed_history = conversation_history[-10:]

    response = ollama.chat(
        model="llama3:8b",
        messages=trimmed_history,
        options={
            "temperature": 0.2,
            "top_p": 0.8,
            "repeat_penalty": 1.2,
            "num_predict": 80
        }
    )

    reply = response["message"]["content"].strip()

    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    return reply


def main() -> None:
    print("=" * 60)
    print(" AI Chatbot (LLaMA 3 + Web Search)")
    print(" Type 'exit' to quit.")
    print("=" * 60)

    conversation_history = [create_system_message()]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        try:
            reply = chat(conversation_history, user_input)
            print(f"\nAssistant: {reply}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
