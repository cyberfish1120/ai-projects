import requests
import argparse
import os

def download_resource(server_url, resource_path, save_path=None):
    """
    从服务器下载资源
    
    参数:
        server_url: 服务器基础URL (例如: http://example.com:8000)
        resource_path: 要获取的资源路径 (例如: /file.txt)
        save_path: 保存文件的路径，默认为当前目录下的资源文件名
    """
    try:
        # 构建完整的请求URL
        url = f"{server_url}{resource_path}"
        
        # 发送GET请求
        print(f"正在请求: {url}")
        response = requests.get(url, timeout=10)
        
        # 检查响应状态码
        response.raise_for_status()  # 如果状态码不是200，会抛出HTTPError
        
        # 确定保存路径
        if not save_path:
            # 从URL中提取文件名
            file_name = resource_path.split('/')[-1]
            if not file_name:  # 如果是根目录，保存为index.html
                file_name = "index.html"
            save_path = os.path.join(os.getcwd(), file_name)
        
        # 保存文件
        with open(save_path, 'wb') as file:
            file.write(response.content)
        
        print(f"资源已成功保存到: {save_path}")
        print(f"文件大小: {len(response.content)} 字节")
        print(f"响应状态码: {response.status_code}")
        
        # 如果是文本类型，显示前100个字符
        content_type = response.headers.get('content-type', '')
        if 'text' in content_type:
            print("\n内容预览:")
            print(response.text[:100] + ('...' if len(response.text) > 100 else ''))
            
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP错误: {e}")
    except requests.exceptions.ConnectionError:
        print("连接错误: 无法连接到服务器，请检查服务器地址和端口")
    except requests.exceptions.Timeout:
        print("超时错误: 连接服务器超时")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        
    return False

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='HTTP客户端，用于获取服务器资源')
    parser.add_argument('server', help='服务器URL (例如: http://your-server-ip:8000)')
    parser.add_argument('resource', help='要获取的资源路径 (例如: /file.txt 或 / 表示文件列表)')
    parser.add_argument('-o', '--output', help='保存文件的路径')
    
    args = parser.parse_args()
    
    # 调用下载函数
    download_resource(args.server, args.resource, args.output)
    