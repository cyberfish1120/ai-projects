import os
import json
from config import abstract_prompt
from volcenginesdkarkruntime import Ark

def ai_process(crawed_papers):
    # 从环境变量中读取您的方舟API Key
    # linux: export ARK_API_KEY="<ARK_API_KEY>"
    # windows: $env:ARK_API_KEY = "<ARK_API_KEY>"
    client = Ark(api_key=os.environ.get("ARK_API_KEY"))

    content = {'title': crawed_papers['title'], 'abstract': crawed_papers['abstract']}
    papers_prompt =  abstract_prompt.replace('{{text}}', json.dumps(content)) 

    completion = client.chat.completions.create(
        # 替换 <Model>为 Model ID
        model="doubao-1.5-thinking-pro-250415",
        messages=[
            {"role": "user", "content": papers_prompt}
        ]
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    client = Ark(api_key=os.environ.get("ARK_API_KEY"))

    completion = client.chat.completions.create(
        model="doubao-1.5-pro-32k-250115",
        messages=[
            {"role": "user", "content": "You are a helpful assistant."}
        ]
    )
    print(completion.choices[0].message.content)