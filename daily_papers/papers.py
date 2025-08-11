import requests
import feedparser
import time
import os
import json
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional

class ArxivPaperCrawler:
    """arXiv论文爬取器"""
    
    def __init__(self, categories: List[str], output_dir: str = "papers"):
        """
        初始化爬取器
        
        Args:
            categories: 感兴趣的论文类别列表，如 ["cs.CV", "cs.LG"]
            output_dir: 保存论文数据的目录
        """
        self.base_url = "http://export.arxiv.org/api/query"
        self.categories = categories
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_daily_papers(self, max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        获取今日新增的论文
        
        Args:
            max_results: 最多获取的论文数量
            
        Returns:
            论文信息列表，每个元素是包含论文详情的字典
        """
        today = datetime.now().date() - timedelta(days=7)
        yesterday = today - timedelta(days=1)
        
        
        # 构建查询条件
        category_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
        date_query = f"submittedDate:[{yesterday.strftime('%Y%m%d')}0000 TO {today.strftime('%Y%m%d')}2359]"
        query = f"({category_query}) AND {date_query}"
        
        # 构建API请求参数
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        papers = []
        try:
            print(f"开始爬取 arXiv 论文，查询条件: {query}")
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            # 解析RSS feed
            feed = feedparser.parse(response.text)
            
            for entry in feed.entries:
                paper = {
                    'id': entry.id.split('/abs/')[-1],
                    'title': entry.title,
                    'authors': [author.name for author in entry.authors],
                    'categories': entry.tags[0]['term'],
                    'abstract': entry.summary,
                    'published': entry.published,
                    'updated': entry.updated,
                    'link': entry.link,
                    'pdf_link': entry.link.replace('abs', 'pdf')
                }
                papers.append(paper)
            
            print(f"成功获取 {len(papers)} 篇论文")
        except Exception as e:
            print(f"爬取论文时出错: {e}")
            raise
        
        return papers

    def run(self):
        """运行爬取器"""
        try:
            papers = self.get_daily_papers()
            if papers:
                print(f"今日爬取完成，共获取 {len(papers)} 篇论文")
                return papers
            else:
                print("今日没有找到符合条件的论文")
                return '今日没有新发布的论文，休息一下吧~'
        except Exception as e:
            print(f"爬取过程中发生致命错误: {e}")
            return f"爬取过程中发生错误: {e}"

if __name__ == '__main__':
    # 设置感兴趣的论文类别
    categories = [
        "cs.AI",  # 人工智能
        # "cs.CL",  # 计算语言学
        # "cs.CV",  # 计算机视觉
        # "cs.LG",  # 机器学习
        # "stat.ML"  # 统计机器学习
    ]
    crawler = ArxivPaperCrawler(categories)
    papers = crawler.run()
