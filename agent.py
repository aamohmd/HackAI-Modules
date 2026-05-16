import os

# Set environment variables in the Python process so LangChain can access them
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "--"
os.environ["LANGSMITH_PROJECT"] = "hack_ai"

print("LangSmith tracing environment variables set.")

from langchain_groq import ChatGroq

model = "openai/gpt-oss-120b"
api_key = "--"

chat = ChatGroq(
    model=model,
    api_key=api_key
)

response = chat.invoke(
    [
        {
            "role":"user",
            "content":"what is the capital of Morocco?",
        }
    ]
).content

print(response)


from langchain.tools import tool

@tool
def calculator_tool(expression:str) -> str:
  "A calculator that can perform basic arithmetic operations"
  try:
      result = eval(expression)
      return str(result)
  except Exception as e:
      return f"Error: {str(e)}"

print(calculator_tool.run("2 + 2 * 3"))

import requests

@tool
def city_weather(city: str) -> str:
    # the tool should have a clear name. and a string documnetation.
    # string documentation has description of the tool and its inputs.
    """
    Get the weather for a given city.
    Args:
        city: The name of the city.
    """
    url = f"https://wttr.in/{city}?format=3"
    
    # Grab the text and return it immediately
    return requests.get(url).text

# test city_weather
tool_output = city_weather.run("Benguerir")
print(tool_output)


from langchain_community.tools import DuckDuckGoSearchRun

# Create a DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun()
# Test Tool
tool_output = search_tool.invoke("What is the capital of Morocco?")
print(tool_output)

from langchain_community.tools.arxiv.tool import ArxivQueryRun

arxiv_tool = ArxivQueryRun()
tool_output = arxiv_tool.run("1706.03762")
print("\n", tool_output)

from langchain.agents import create_agent

tools=[city_weather,search_tool,calculator_tool,arxiv_tool]
agent=create_agent(
    model=chat,
    tools=tools
)

query="""What's the paper 1706.03762 about,
and who is Noam Shazeer?
Also, what is the weather in Benguerir?
And what is 2 + 2 * 3?"""
response=agent.invoke(
    {
    "messages": [{"role": "user", "content": query}]}
)

print(response["messages"][-1].content)

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Short-term memory storage
memory = InMemorySaver()

agent = create_agent(
    model=chat,
    tools=[city_weather, search_tool, calculator_tool, arxiv_tool],
    checkpointer=memory
)

config = {
    "configurable": {
        "thread_id": "1337" # set your id here
    }
}

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "hello my name adil"}
        ]
    },
    config=config
)
print(response["messages"][-1].content)

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "what is my name?"}
        ]
    },
    config=config
)
print(response["messages"][-1].content)

