import requests
import json
from typing import List, Dict, Optional, Tuple, Union

class Ollama:
    """
    用于与本地Ollama服务交互的客户端类，专门调用 huihui_ai/hunyuan-mt-abliterated:latest 模型
    """
    
    _base_url: str = "http://127.0.0.1:11434"  # Ollama 默认地址
    _model: str = "huihui_ai/hunyuan-mt-abliterated:latest"
    _timeout: int = 300  # 超时时间（秒），翻译可能较耗时

    def __init__(self, base_url: str = None, model: str = None, timeout: int = None):
        """
        初始化Ollama客户端
        
        :param base_url: Ollama服务地址，默认为 http://127.0.0.1:11434
        :param model: 要使用的模型名称
        :param timeout: 请求超时时间
        """
        if base_url:
            self._base_url = base_url.rstrip('/')  # 移除末尾斜杠
        if model:
            self._model = model
        if timeout:
            self._timeout = timeout

    def __send_request(self, 
                      messages: List[Dict[str, str]], 
                      stream: bool = False,
                      **kwargs) -> requests.Response:
        """
        向Ollama API发送请求（私有方法）
        
        :param messages: 消息列表
        :param stream: 是否使用流式传输
        :return: 响应对象
        """
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.1,  # 低温度保证翻译准确性
                "top_p": 0.9,
                # 可以根据需要添加其他模型参数
            }
        }
        # 合并额外的参数
        payload["options"].update(kwargs)
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                timeout=self._timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()  # 如果状态码不是200，抛出异常
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"请求Ollama API失败: {str(e)}")

    @staticmethod
    def __get_session(session_id: str, message: str, session_cache: dict) -> List[dict]:
        """
        获取或创建会话上下文
        """
        session = session_cache.get(session_id, [])
        
        # 添加系统提示词（如果是新会话）
        if not session:
            session.append({
                "role": "system",
                "content": "你是一个专业的翻译专家，请准确流畅地进行中英互译。"
            })
        
        # 添加用户消息
        session.append({
            "role": "user",
            "content": message
        })
        
        return session

    @staticmethod
    def __save_session(session_id: str, assistant_message: str, session_cache: dict):
        """
        保存助手回复到会话上下文
        """
        session = session_cache.get(session_id, [])
        session.append({
            "role": "assistant",
            "content": assistant_message
        })
        session_cache[session_id] = session

    def translate_text(self, 
                      text: str, 
                      target_lang: str = "zh",
                      context: str = None,
                      session_id: str = None,
                      session_cache: dict = None,
                      **kwargs) -> Tuple[bool, str]:
        """
        翻译文本
        
        :param text: 要翻译的文本
        :param target_lang: 目标语言，默认为中文(zh)
        :param context: 翻译上下文信息
        :param session_id: 会话ID，用于维护多轮对话上下文
        :param session_cache: 会话缓存字典
        :return: (成功与否, 翻译结果或错误信息)
        """
        # 构建系统提示词
        system_prompt = f"""你是一位专业的翻译专家，请严格遵循以下规则：
1. 将文本精准翻译为{target_lang}，保持原文本意
2. 使用自然的口语化表达，符合目标语言的阅读习惯
3. 结合上下文语境，保持术语和风格的一致性
4. 只输出译文，不要添加任何解释、说明或额外内容"""

        # 构建用户消息
        user_message = f"需要翻译的文本：\n{text}"
        if context:
            user_message = f"翻译上下文：\n{context}\n\n{user_message}"

        # 准备消息列表
        messages = []
        
        # 添加系统提示
        messages.append({"role": "system", "content": system_prompt})
        
        # 处理会话上下文
        if session_id and session_cache is not None:
            messages = self.__get_session(session_id, user_message, session_cache)
        else:
            messages.append({"role": "user", "content": user_message})

        try:
            # 发送请求到Ollama
            response = self.__send_request(messages, **kwargs)
            response_data = response.json()
            
            # 提取翻译结果
            if 'message' in response_data and 'content' in response_data['message']:
                translated_text = response_data['message']['content'].strip()
                
                # 保存到会话历史（如果启用了会话）
                if session_id and session_cache is not None:
                    self.__save_session(session_id, translated_text, session_cache)
                
                return True, translated_text
            else:
                return False, "API响应格式异常"
                
        except Exception as e:
            return False, f"翻译过程中出错: {str(e)}"

    def clear_session(self, session_id: str, session_cache: dict):
        """
        清除指定会话的上下文
        """
        if session_id in session_cache:
            del session_cache[session_id]

    def list_models(self) -> Tuple[bool, List[str]]:
        """
        获取可用的模型列表
        """
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            return True, models
        except Exception as e:
            return False, [f"获取模型列表失败: {str(e)}"]

    def check_health(self) -> bool:
        """
        检查Ollama服务是否健康
        """
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# 使用示例
if __name__ == "__main__":
    # 初始化翻译器
    translator = Ollama()
    
    # 检查服务是否正常
    if translator.check_health():
        print("✅ Ollama服务正常运行")
        
        # 获取模型列表（可选）
        success, models = translator.list_models()
        if success:
            print(f"可用模型: {models}")
    else:
        print("❌ Ollama服务未启动或无法连接")
        exit(1)
    
    # 创建一个简单的内存会话缓存
    session_cache = {}
    session_id = "user_123"
    
    # 示例1：简单翻译
    text_to_translate = "Hello, world! This is a test translation."
    success, result = translator.translate_text(text_to_translate)
    
    if success:
        print(f"📝 原文: {text_to_translate}")
        print(f"🇨🇳 译文: {result}")
    else:
        print(f"翻译失败: {result}")
    
    # 示例2：带会话的翻译（保持上下文）
    print("\n--- 带会话的翻译 ---")
    text1 = "What is the weather like today?"
    success, result1 = translator.translate_text(text1, session_id=session_id, session_cache=session_cache)
    print(f"Q: {text1}")
    print(f"A: {result1}")
    
    # 第二句话会继承之前的上下文
    text2 = "And tomorrow?"
    success, result2 = translator.translate_text(text2, session_id=session_id, session_cache=session_cache)
    print(f"Q: {text2}")
    print(f"A: {result2}")
    
    # 清理会话
    translator.clear_session(session_id, session_cache)