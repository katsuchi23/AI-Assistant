from .components import ChatBot

if __name__ == "__main__":
    try:
        bot = ChatBot()
        bot.chat()
    except Exception as e:
        print(f"Error: {e}")