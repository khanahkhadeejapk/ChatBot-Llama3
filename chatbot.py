"""
AI Chatbot using Ollama with LLaMA 3
=====================================
A terminal-based chatbot that uses the Ollama framework
to run the LLaMA 3 large language model locally.
"""

import ollama
import sys


def create_system_message() -> dict:
    """Create the system message that defines the chatbot's behaviour."""
    return {
        "role": "system",
        "content": (
            "You are a helpful, friendly AI assistant. "
            "Provide clear, concise, and accurate answers. "
            "If you are unsure about something, say so honestly."
        ),
    }


def chat(conversation_history: list[dict]) -> str:
    """
    Send the full conversation history to the LLaMA 3 model via Ollama
    and return the assistant's reply.

    Parameters
    ----------
    conversation_history : list[dict]
        A list of message dicts with 'role' and 'content' keys.

    Returns
    -------
    str
        The model's generated response text.
    """
    response = ollama.chat(
        model="llama3",
        messages=conversation_history,
    )
    return response["message"]["content"]


def main() -> None:
    """Run the interactive chatbot loop."""

    print("=" * 60)
    print("  AI Chatbot powered by LLaMA 3 (via Ollama)")
    print("  Type 'exit' to quit.")
    print("=" * 60)

    # Initialise conversation history with system prompt
    conversation_history: list[dict] = [create_system_message()]

    while True:
        # --- Get user input ---
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

        # --- Append user message to history ---
        conversation_history.append({"role": "user", "content": user_input})

        # --- Get model response ---
        try:
            reply = chat(conversation_history)
        except Exception as e:
            print(f"\n[Error] Failed to get a response from the model: {e}")
            # Remove the failed user message so history stays consistent
            conversation_history.pop()
            continue

        # --- Append assistant reply and display ---
        conversation_history.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}")


if __name__ == "__main__":
    main()