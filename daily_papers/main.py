import json
import time
from datetime import datetime
from papers import ArxivPaperCrawler
from ai import ai_process
from email_send import send_email
from pdf_read import paper_read

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
receiver = "zqzhao1221@outlook.com"
receiver = "1028755879@qq.com"
password = "endcdkgcxzavbgca"  # 或应用专用密码

# 创建并运行爬取器
crawler = ArxivPaperCrawler(categories)
while True:
    if datetime.now().hour != 6:
        time.sleep(1200)
        continue
    papers = crawler.run()

    output = []
    if type(papers)==list:
        papers = [paper for paper in papers if paper['categories'] in categories]
        
        print(f'符合条件的论文共{len(papers)}篇')
        # papers = papers[:50]
        for paper in papers:
            res = ai_process(paper)
            try:
                res = json.loads(res)
            except:
                res = {
                    'title': paper['title'],
                    'abstract': paper['abstract']
                }
            # coze_analyse = paper_read(paper['pdf_link'])
            # try:
            #     coze_analyse = json.loads(json.loads(coze_analyse)['output'])
            # except:
            #     coze_analyse = {
            #         '主要贡献': '',
            #         '研究方法描述': '',
            #         '实验结果解释': '',
            #     }
            output.append({
                'title': res['title'],
                'abstract': res['abstract'],
                'authors': paper['authors'],
                # 'contributions': coze_analyse['主要贡献'],
                # 'methods': coze_analyse['研究方法描述'],
                # 'results': coze_analyse['实验结果解释'],
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
        receiver_email=receiver,
        subject=f"今日论文更新【新上传{paper_count}篇】 ({datetime.now().strftime('%Y-%m-%d')})",
        papers=output,
        smtp_server="smtp.qq.com",
        smtp_port=587,
        username=sender,
        password=password
    )    
    time.sleep(3600)