from src.response_generator import generate_response

def simulate_chat():
    print("Start chatting with the bot (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        print("Bot:", generate_response(user_input))
