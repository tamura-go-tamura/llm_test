import os
import json
import os\


from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import BraveSearch
# Webページ読み込み用関数。渡されたURLの本文を返却する
from langchain_community.document_loaders import WebBaseLoader

from datetime import datetime
import time



def web_page_reader(url: str) -> str:
    loader = WebBaseLoader(url)
    content = ','.join([i.page_content.replace('\n','').replace('\t','') for i  in loader.load()])
    return content

# 使用可能なツールと説明
tools = [
    BraveSearch.from_api_key(api_key=os.environ['BRAVE_API_KEY'], search_kwargs={"count": 3}),
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

def handler(event, context):
    # イベントとコンテキストの内容を出力
    print( "event:" , event )
    print( "context:", context )

    # パラメータを取得する
    latitude = float(event.get('queryStringParameters').get('latitude'))
    longitude = float(event.get('queryStringParameters').get('longitude'))

    while True:
        try:
            response = get_llm_json(latitude=latitude, longitude=longitude)
            break
        except:
            time.sleep(5)

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }



def get_llm_json(latitude: float, longitude : float):
    result = agent({"input": f"今日は{datetime.today()}です。今日ならではの、経度:{longitude}、緯度:{latitude}周辺の" + 
                """
                ワクワクするようなイチオシ情報を教えてください！また出勤途中のサラリーマンに激励の言葉を与えてください。情報源となった、リンクも一緒に提示してください。
                回答のフォーマットは以下のjson形式で与えてください。json以外の文字列は一切与えてはいけません。また、actionを行なっていいのは3回までです。
                {
                    "comment": "ここに激励のコメントを書いてください! example: 今日はevent1のような楽しいイベントが開催されているみたいです。仕事帰りに寄ってってはいかがでしょうか？今日も頑張ってください！",
                    "events":[
                        {
                            "name" : "event1",
                            "url" : "https://..."
                        }
                    ]
                }
                """}, include_run_info=True)

    last_index = result['output'].rfind('}')
    print(result['output'])
    return json.loads(result['output'][:last_index+1]) 