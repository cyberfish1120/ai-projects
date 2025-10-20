import re
import os
import json
import httpx
import vertexai
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.oauth2 import service_account
from datetime import datetime
from cozepy import COZE_CN_BASE_URL, Coze, TokenAuth
from time import sleep
from dotenv import load_dotenv

load_dotenv()

class PaperAnalyse(BaseModel):
    motivation: str
    method: str
    conclusion: str

def gemini_pdf_trans(pdf_url):
    with open('http_download/pdf_trans/prompt.txt', encoding='utf8') as fr:
        prompt = fr.read()
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
        system_instruction=prompt,
        response_mime_type="application/json",
        response_schema=PaperAnalyse
    )

    # Retrieve and encode the PDF byte
    doc_data = httpx.get(pdf_url).content
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=model_config,
        contents=[
            types.Part.from_bytes(
                data=doc_data,
                mime_type='application/pdf',
            ),
            prompt])
    return response.parsed.motivation, response.parsed.method, response.parsed.conclusion

def coze_pdf_trans(pdf_url):
    # 扣子配置
    coze_api_token = os.environ.get('coze_api_token')
    coze_api_base = COZE_CN_BASE_URL
    coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=coze_api_base)
    workflow_id = '7517842621692968969'
    max_trans_n = 10
    cur_trans_n = 0
    while cur_trans_n < max_trans_n:
        cur_trans_n += 1
        try:
            workflow = coze.workflows.runs.create(
                workflow_id=workflow_id,
                parameters={
                    'input': pdf_url
                }
                # is_async=True
            )
            # pattern = r'```json\s*(.*?)\s*```'
            # match = re.search(pattern, workflow.data)motivation
            # coze_analyse = json.loads(match.group(1))
            coze_analyse = json.loads(json.loads(workflow.data)['output'])
            return coze_analyse['motivation'], coze_analyse['method'], coze_analyse['conclusion']
        except Exception as e:
            print(f'详细解析出错，正在重试{cur_trans_n}：{e}')

def paper_analyse(params):
    # 获取标题和摘要
    paper_date = params.get('date', [0])[0]
    pdf_url = params.get('pdf_link', [0])[0]
    id = pdf_url.split('/')[-1]
    with open(f'/root/每日论文/paper_data/{paper_date}/{id}.json') as fr:
        paper_info = json.load(fr)
    title = paper_info['标题']
    abstract = paper_info['摘要']
    if '动机' in paper_info:
        paper_analyse = {
            '动机': paper_info['motivation'],
            '方法': paper_info['method'],
            '结论': paper_info['conclusion'],
        }
        print('直接获取详情')
    else:
        try:
            print('gemini生成中...')
            motivation, method, conclusion = gemini_pdf_trans()
            paper_analyse = {
                '动机': motivation,
                '方法': method,
                '结论': conclusion,
            }
        except Exception as e_gemini:
            print(f'gemini调用接口出错：{e_gemini}，尝试使用扣子解析pdf文档')
            try:
                motivation, method, conclusion = coze_pdf_trans()
                paper_analyse = {
                    '动机': motivation,
                    '方法': method,
                    '结论': conclusion,
                }
            except Exception as e_coze:
                print(f'coze api token过期或余额不足，请更新！{e_coze}')
                paper_analyse = {
                    '动机': 'coze api token过期或余额不足！',
                    '方法': 'coze api token过期或余额不足！',
                    '结论': 'coze api token过期或余额不足！',
                }
        
    with open(f'/root/每日论文/paper_data/{paper_date}/{id}.json', 'w') as fw:
        json.dump(paper_info, fw)
    
    with open('/root/http_download/pdf_trans/index.html') as fr:
        html_template = fr.read()
    
    html_content = html_template.replace('{title}', title) \
    .replace('{abstract}', abstract) \
    .replace('{motivation}', paper_analyse['动机']) \
    .replace('{method}', paper_analyse['方法']) \
    .replace('{conclusion}', paper_analyse['结论']) \
    .replace('{pdf_link}', pdf_url)
    return html_content

if __name__ == '__main__':
    # a = gemini_pdf_trans('https://arxiv.org/pdf/1810.04805')
    # b = coze_pdf_trans('https://arxiv.org/pdf/1810.04805')
    print()