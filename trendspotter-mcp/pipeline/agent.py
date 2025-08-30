# pipeline/agent.py
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from mcp import MCPClient
from .text_analysis import analyze_text

llm = ChatOpenAI(model="gpt-4")

# Connect to MCP server
twitter_mcp = MCPClient("http://localhost:4000/twitter")

def get_twitter_trends(query):
    data = twitter_mcp.call("get_trends", {"query": query})
    return analyze_text(data)  # Sentiment + keyword extraction

twitter_tool = Tool(
    name="TwitterTrends",
    func=get_twitter_trends,
    description="Fetch and analyze Twitter hashtags"
)

agent = initialize_agent([twitter_tool], llm, agent_type="zero-shot-react-description")

if __name__ == "__main__":
    print(agent.run("Find rising beauty trends among Gen Z"))
