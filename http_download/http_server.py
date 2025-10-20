import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs
from cozepy import COZE_CN_BASE_URL, Coze, TokenAuth
from pdf_trans.pdf_trans import pdf_trans
from time import sleep

# 定义业务逻辑处理函数
def process_business_logic(params, parsed_data):
    """处理业务逻辑的示例函数"""
    # 这里可以根据需要实现任何业务逻辑
    operation = params.get('operation', [''])[0]
    result = {"status": "success", "message": "业务逻辑执行成功"}
    # pdf_trans业务
    if operation == 'pdf_trans':
        # post请求
        # try:
        #     title = str(parsed_data).split('title')[1].split('----')[0].replace(r'\r\n', '')
        #     abstract = str(parsed_data).split('abstract')[1].split('----')[0].replace(r'\r\n', '')
        # except:
        #     title = '解析错误'
        #     abstract = '解析错误'
        # get请求
        return pdf_trans(params)
    else:
        result["status"] = "warning"
        result["message"] = f"未知操作: {operation}，使用默认响应"
    
    return result

# 自定义请求处理器
class CustomHandler(http.server.BaseHTTPRequestHandler):
    # 处理GET请求
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)
        
        try:
            # 获取内容长度
            content_length = int(self.headers['Content-Length'])
            # 读取POST数据
            post_data = self.rfile.read(content_length)
            # 解析表单数据
            parsed_data = parse_qs(post_data.decode('utf-8'))
        except:
            parsed_data = ''
        
        # 处理文件请求
        if path.startswith('/file'):
            # 提取文件名 (去掉/file前缀)
            file_path = path[len('/file'):]
            # 安全处理：防止访问上级目录
            file_path = os.path.normpath(file_path).lstrip('/')
            
            # 如果未指定文件，返回可用文件列表
            if not file_path:
                try:
                    files = os.listdir('./resources')
                    # 构建HTML响应
                    html = "<html><head><title>File List</title></head>"
                    html += "<body><h1>Available Files</h1><ul>"
                    for file in files:
                        html += f"<li><a href='/file/{file}'>{file}</a></li>"
                    html += "</ul></body></html>"
                    
                    # 发送响应
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.send_header("Content-length", str(len(html.encode())))
                    self.end_headers()
                    self.wfile.write(html.encode())
                except Exception as e:
                    self.send_error(500, f"Server error: {str(e)}")
                return
            
            # 检查文件是否存在
            file_path = './resources/' + file_path
            if os.path.exists(file_path) and os.path.isfile(file_path):
                try:
                    # 读取文件内容
                    with open(file_path, 'rb') as file:
                        content = file.read()
                    
                    # 发送响应
                    self.send_response(200)
                    self.send_header('Content-type', 'application/octet-stream')
                    self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(file_path)}"')
                    self.send_header('Content-Length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {
                        "status": "error",
                        "message": f"读取文件失败: {str(e)}"
                    }
                    self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "status": "error",
                    "message": f"文件不存在: {file_path}"
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
        
        # 处理API请求
        elif path.startswith('/api'):
            try:
                # 执行业务逻辑
                result = process_business_logic(query_params, parsed_data)
                
                # 发送响应
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.send_header('Content-length', str(len(result.encode())))
                self.end_headers()
                self.wfile.write(result.encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "status": "error",
                    "message": f"API处理失败: {str(e)}"
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
        
        # 处理未知路径
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "error",
                "message": f"未知路径: {path}"
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))

# 启动服务器
def run_server(host='0.0.0.0', port=8000):
    server_address = (host, port)
    httpd = socketserver.TCPServer(server_address, CustomHandler)
    print(f"服务器启动，监听 {host}:{port}")
    print("支持的端点:")
    print("  - /file 获取文件列表")
    print("  - /file/filename 获取指定文件")
    print("  - /api 执行业务逻辑，支持参数 operation=add&a=1&b=2")
    while True:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器正在关闭...")
            httpd.shutdown()
            break
        except Exception as e:
            print(f'服务出错：{e}，两分钟后重试')
            sleep(120)

if __name__ == '__main__':
    run_server()
