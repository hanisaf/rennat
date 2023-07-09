from langchain.agents import tool
from datetime import date
from util import Util
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
import langchain

@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())

if __name__ == "__main__":
    print(time(""))
    
    langchain.debug=True
    llm = Util.get_llm(None)
    tools = load_tools(["llm-math","wikipedia"], llm=llm)
    agent= initialize_agent(
     [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
    try:
        result = agent("whats the date today?") 
    except: 
        print("exception on external access")