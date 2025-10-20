import os
import json
import vertexai
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.oauth2 import service_account
from config import abstract_prompt
from volcenginesdkarkruntime import Ark
from dotenv import load_dotenv

class TransResponse(BaseModel):
    title_cn: str
    abstract_cn: str

load_dotenv()
vertexai.init(project=os.environ.get('g_project'), location=os.environ.get('g_location'))
credentials = service_account.Credentials.from_service_account_file(
    os.environ.get('g_file'),
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)
client = genai.Client(
    vertexai = True,
    project=os.environ.get('g_project'),
    location=os.environ.get('g_location'),
    credentials=credentials
)
model_config = types.GenerateContentConfig(
    system_instruction=abstract_prompt.replace('{{text}}', ''),
    response_mime_type="application/json",
    response_schema=TransResponse
)
# 采用gemini api翻译标题和摘要
def trans_gemini(crawed_papers):
    contents = f"title: {crawed_papers['title']}\n\nabstract: {crawed_papers['abstract']}"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=model_config,
        contents=contents
    )

    return response.parsed

# 采用豆包api翻译标题和摘要
def trans_doubao(crawed_papers):
    # 从环境变量中读取您的方舟API Key
    # linux: export ARK_API_KEY="<ARK_API_KEY>"
    # windows: $env:ARK_API_KEY = "<ARK_API_KEY>"
    client = Ark(api_key=os.environ.get("ARK_API_KEY"))

    content = {'title': crawed_papers['title'], 'abstract': crawed_papers['abstract']}
    papers_prompt =  abstract_prompt.replace('{{text}}', json.dumps(content)) 

    completion = client.beta.chat.completions.parse(
        # 替换 <Model>为 Model ID
        model="doubao-seed-1-6-flash-250715",
        messages=[
            {"role": "user", "content": papers_prompt}
        ],
        response_format=TransResponse
    )
    return completion.choices[0].message.parsed

if __name__ == '__main__':
    client = Ark(api_key=os.environ.get("ARK_API_KEY"))

    completion = client.chat.completions.create(
        model="doubao-seed-1-6-flash-250715",
        messages=[
            {"role": "user", "content": "You are a helpful assistant."}
        ],
        response_format=TransResponse
    )
    print(completion.choices[0].message.content)