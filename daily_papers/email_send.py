import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from datetime import datetime

def format_papers_to_html(papers: list) -> str:
    """将详细论文列表格式化为HTML"""
    if type(papers)!=list:
        return f"<p>{papers}</p>"
    
    # 构建HTML头部和样式
    html = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }
            .paper-container { margin-bottom: 30px; padding: 15px; border-radius: 8px; background-color: #fff; 
                              box-shadow: 0 2px 5px rgba(0,0,0,0.1); transition: transform 0.2s; }
            .paper-container:hover { transform: translateY(-2px); }
            .paper-header { margin-bottom: 10px; }
            .paper-title { font-size: 18px; font-weight: bold; margin-bottom: 5px; }
            .paper-title a { color: #1a73e8; text-decoration: none; }
            .paper-title a:hover { text-decoration: underline; }
            .paper-meta { font-size: 13px; color: #666; margin-bottom: 10px; }
            .paper-section { margin-bottom: 15px; }
            .section-title { font-size: 15px; font-weight: bold; color: #444; margin-bottom: 5px; }
            .section-content { font-size: 14px; line-height: 1.5; }
            .divider { border-top: 1px solid #eee; margin: 20px 0; }
            .timestamp { font-size: 12px; color: #888; text-align: right; }
        </style>
    </head>
    <body>
        <h2 style="color: #333; margin-bottom: 20px;">今日论文更新</h2>
    """
    current_time = datetime.now().strftime("%Y%m%d")
    # 添加每篇论文
    for idx, paper in enumerate(papers):
        # 确保所有字段都存在
        title = paper.get('title', '无标题')
        authors = paper.get('authors', [])
        abstract = paper.get('abstract', '无摘要')
        contributions = paper.get('contributions', '未提供主要贡献')
        methods = paper.get('methods', '未提供研究方法')
        results = paper.get('results', '未提供实验结果')
        published = paper.get('published', '未知')
        updated = paper.get('updated', '未知')
        link = paper.get('link', '#')
        pdf_link = paper.get('pdf_link', '#')
        
        # 格式化作者列表
        authors_str = ', '.join(authors) if isinstance(authors, list) else authors
        # 为每个链接创建唯一ID
        link_id = f"link-{idx}"
        # 为每个链接创建一个表单，用于POST请求
        form_id = f"form-{idx}"
        # 添加论文卡片
        html += f"""
        <div class="paper-container">
            <div class="paper-header">
                <div class="paper-title"><a href="{link}" target="_blank">{title}</a></div>
                <div class="paper-meta">
                    <span style="font-weight: bold;">作者:</span> {authors_str} | 
                    <span style="font-weight: bold;">发布时间:</span> {published} | 
                    <span style="font-weight: bold;">更新时间:</span> {updated} |
                </div>
                <div>pdf链接：<a href="{pdf_link}" target="_blank">{pdf_link}</a></div>
                <div>加载详情：<a href="http://www.cyberfish.fun:8000/api/?operation=pdf_trans&pdf_link={pdf_link}&date={current_time}" target="_blank">www.cyberfish.fun/papers/{idx}</a></div>
            </div>
            
            <div class="paper-section">
                <div class="section-title">摘要</div>
                <div class="section-content">{abstract}</div>
            </div>
        </div>
        
        {'' if idx == len(papers) - 1 else '<div class="divider"></div>'}
        """
    
    html += f"""
        <div class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        <p style="font-size: 13px; color: #888; margin-top: 20px;">此邮件由自动系统生成，请勿回复。</p>
    </body>
    </html>
    """
    
    return html

def send_email(
    sender_email: str,
    receiver_emails: list,
    subject: str,
    papers: list,
    smtp_server: str = "smtp.qq.com",
    smtp_port: int = 587,
    username: str = None,
    password: str = None
):
    """
    将论文列表以HTML格式发送邮件
    
    Args:
        sender_email: 发件人邮箱
        receiver_emails: 收件人邮箱
        subject: 邮件主题
        papers: 论文信息列表
        smtp_server: SMTP服务器地址
        smtp_port: SMTP服务器端口
        username: 邮箱用户名
        password: 邮箱密码或授权码
    """
    # 创建邮件对象
    message = MIMEMultipart()
    message["From"] = Header(sender_email)    
    message["Subject"] = Header(subject, "utf-8")
    
    # 格式化论文为HTML
    html_content = format_papers_to_html(papers)
    
    # 添加HTML内容
    message.attach(MIMEText(html_content, "html", "utf-8"))
    
    try:
        for receiver_email in receiver_emails:
            # 连接SMTP服务器
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  # 启用TLS加密
            server.login(username or sender_email, password)
            
            # 发送邮件
            message["To"] = Header(receiver_email, "utf-8")
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("邮件发送成功")
    except Exception as e:
        print(f"邮件发送失败: {e}")
    finally:
        server.quit()  # 关闭连接

if __name__ == "__main__":
    # 示例论文数据
    sample_papers = [
        {
            'title': 'Attention Is All You Need',
            'authors': ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
            'abstract': 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms...',
            'contributions': '1. 提出Transformer架构；2. 证明self-attention机制的有效性；3. 在机器翻译任务上取得SOTA结果。',
            'methods': '1. 多头自注意力机制；2. 位置编码；3. 残差连接和层归一化。',
            'results': '在WMT 2014 English-to-German翻译任务上达到28.4 BLEU，在WMT 2014 English-to-French翻译任务上达到41.8 BLEU。',
            'published': '2023-06-15',
            'updated': '2023-06-15',
            'link': 'https://arxiv.org/abs/1706.03762'
        },
        {
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
            'authors': ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee'],
            'abstract': 'We introduce a new language representation model called BERT...',
            'contributions': '1. 提出双向Transformer预训练方法；2. 引入Masked LM和Next Sentence Prediction任务；3. 在11个NLP任务上取得SOTA结果。',
            'methods': '1. 基于Transformer的双向编码器；2. 大规模无监督预训练；3. 微调策略应用于下游任务。',
            'results': '在GLUE基准测试中取得80.5分，在SQuAD v1.1问答任务上取得93.2 F1分数，在SQuAD v2.0上取得83.1 F1分数。',
            'published': '2023-06-14',
            'updated': '2023-06-14',
            'link': 'https://arxiv.org/abs/1810.04805'
        }
    ]
    
    # 配置邮件参数（请替换为实际信息）
    sender = "2143976877@qq.com"
    receiver = "zqzhao1221@outlook.com"
    receiver = "1028755879@qq.com"
    password = "endcdkgcxzavbgca"  # 或应用专用密码
    
    # 发送邮件
    send_email(
        sender_email=sender,
        receiver_email=receiver,
        subject=f"今日论文更新 ({datetime.now().strftime('%Y-%m-%d')})",
        papers=sample_papers,
        smtp_server="smtp.qq.com",
        smtp_port=587,
        username=sender,
        password=password
    )    