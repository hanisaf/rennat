import sys
from langchain.agents import tool
from datetime import date
from util import Util
from index import Index
from rennat import Rennat
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
import langchain

index : Index = None

@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())

@tool
def answer(question: str) -> str:
    """Answers a question based on the input \
    question. It uses the library of references \
    to extract an answer from the question. \
    It also returns a list of references that \
    were used to answer the question.
    """
    text, papers, sources = index.answer(question)
    res = f"""The answer to the question: {question} is: {text}
    The references used to answer this question are: 
    {papers}"""
    return res

if __name__ == "__main__":
    index_file = sys.argv[1]
    openai_token = sys.argv[2]
    collection_name = sys.argv[3]
    rennat = Rennat(index_file, collection_name, openai_token) 
    rennat.current_collection()
    index = rennat.index

    print(time(""))
    
    langchain.debug=True
    llm = Util.get_llm(None)
    tools = load_tools(["llm-math","wikipedia"], llm=llm)
    agent= initialize_agent(
      [time, answer], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
    try:
        result = agent("what is clan control?") 
    except: 
        print("exception on external access")