import os
from dotenv import load_dotenv

load_dotenv("../.env")
# os.environ['GROQ_API_KEY']=os.getenv['OPENAI_API_KEY']

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.tools import DuckDuckGoSearchRun

# Webページ読み込み用関数。渡されたURLの本文を返却する
from langchain.document_loaders import WebBaseLoader

def web_page_reader(url: str) -> str:
    loader = WebBaseLoader(url)
    content = ','.join([i.page_content.replace('\n','').replace('\t','') for i  in loader.load()])
    return content

# Web検索用ツール
search = DuckDuckGoSearchRun()

# 使用可能なツールと説明
tools = [
    Tool(
        name="duckduckgo-search",
        func=search.run,
        description="このツールはWeb上の最新情報を検索します。引数で検索キーワードを受け取ります。最新情報が必要ない場合はこのツールは使用しません。",
    ),
    Tool(
        name = "WebBaseLoader",
        func=web_page_reader,
        description="このツールは引数でURLを渡された場合に内容をテキストで返却します。引数にはURLの文字列のみを受け付けます。"
    )
]


llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
)

agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent="zero-shot-react-description",
            verbose=True,
        )
agent.agent.llm_chain.prompt.template = (
            "Answers must be given in Japanese and thoughts need to be spelled out in English. You need to use Tool at least once."  # noqa: E501
            + agent.agent.llm_chain.prompt.template
        )
result = agent({"input": "今日の日付ならではの、横浜駅近辺のワクワクするようなイチオシ情報を教えてください！情報源となった、リンクも一緒に提示してください。"}, include_run_info=True)

# Agentの実行
print(result['output'])