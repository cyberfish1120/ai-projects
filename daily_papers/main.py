import json
import time
from datetime import datetime
from papers import ArxivPaperCrawler
from ai import trans_gemini, trans_doubao
from email_send import send_email
from pdf_read import paper_read
from datetime import datetime
import os

# 设置感兴趣的论文类别
categories = [
    "cs.AI",  # 人工智能
    # "cs.CL",  # 计算语言学
    # "cs.CV",  # 计算机视觉
    # "cs.LG",  # 机器学习
    # "stat.ML"  # 统计机器学习
]

# 配置邮件参数
sender = "2143976877@qq.com"
receivers = [
    # "3201520786@qq.com", 
    "1028755879@qq.com"
]
password = "endcdkgcxzavbgca"  # 或应用专用密码

# 创建并运行爬取器
crawler = ArxivPaperCrawler(categories)
max_trans_n = 10
while True:
    try:
        if datetime.now().hour != 15:
            print('未到指定发送时间，静默30分钟')
            time.sleep(1800)
            continue

        papers = crawler.run()
        output = []
        if type(papers)==list:
            papers = [paper for paper in papers if paper['categories'] in categories]
            
            print(f'符合条件的论文共{len(papers)}篇')
            # papers = papers[:4]
            current_time = datetime.now().strftime("%Y%m%d")
            os.makedirs(f'./paper_data/{current_time}', exist_ok=True)
            for paper in papers:
                try: 
                    res = trans_gemini(paper)
                except Exception as e:
                    print(f'gemini翻译出错：{e}\n采用豆包翻译')
                    try:
                        res = trans_doubao(paper)
                    except Exception as e:
                        print(f'豆包翻译出错：{e}')
                    
                paper['标题'] = res.title_cn
                paper['摘要'] = res.abstract_cn
                with open(f'./paper_data/{current_time}/{paper["id"]}.json', 'w') as fw:
                    json.dump(paper, fw, ensure_ascii=False)
                output.append({
                    'title': res['title_cn'],
                    'abstract': res['abstract_cn'],
                    'authors': paper['authors'],
                    'published': paper['published'],
                    'updated': paper['updated'],
                    'link': paper['link'],
                    'pdf_link': paper['pdf_link'],
                })
            paper_count = len(output)
        else:
            output = papers
            paper_count = 0
        # 发送邮件
        send_email(
            sender_email=sender,
            receiver_emails=receivers,
            subject=f"今日论文更新【新上传{paper_count}篇】 ({datetime.now().strftime('%Y-%m-%d')})",
            papers=output,
            smtp_server="smtp.qq.com",
            smtp_port=587,
            username=sender,
            password=password
        )    
        time.sleep(3600)
    except Exception as e:
        print(f'ERROR: {e}，60秒后重试！')
        time.sleep(60)