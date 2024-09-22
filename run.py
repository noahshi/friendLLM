from friend_llm import FriendLLM

MODEL_PATH = "./models/model_8000.pth"

if __name__ == "__main__":
	prompt = input("Enter LLM Prompt: ")

	bot = FriendLLM()
	bot.load_model(MODEL_PATH)
	bot.prompt(prompt)