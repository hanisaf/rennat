from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from util import Util


class ChatBot:

    def __init__(self, name, openai_token=None):
        self.name = name
        self.llm = Util.get_llm(openai_token)
        self.messages = [HumanMessage(content=f"You are {name}")]

    def inform(self, message:str):
        self.messages.append(HumanMessage(content=message))

    def chat(self, message:str):
        self.messages.append(HumanMessage(content=message))
        result = self.llm(self.messages)
        self.messages.append(result)
        return result.content
    
    def reset(self):
        self.messages = self.messages[0:1]

    def __repr__(self):
        return f"ChatBot({self.name})"
    
if __name__ == "__main__":
    bot = ChatBot("a helpful bot")
    while True:
        message = input("You: ")
        response = bot.chat(message)
        print(f"\n{bot.name}: {response}")

    