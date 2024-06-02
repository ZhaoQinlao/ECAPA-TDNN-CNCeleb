from urllib.parse import urlencode
from pydub import AudioSegment
import websocket
import ssl
import base64
import hashlib
import json
import time
import threading
from datetime import datetime
from wsgiref.handlers import format_date_time
from time import mktime
import hmac

# 替换为你的科大讯飞 API 应用 ID、API Key 和 API Secret
APP_ID = '43350a74'
API_KEY = '07becc4453e4c61519995b0d01c41bf1'
API_SECRET = 'MjM2MDUxNDdlYTY2OTk0ZTVmNWJlZmZk'

def get_auth_url():
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(API_SECRET.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            API_KEY, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url

def on_message(ws, message):
    global result_text
    result = json.loads(message)
    if 'data' in result and 'result' in result['data']:
        for w in result['data']['result']['ws']:
            result_text += w['cw'][0]['w']
        print(result_text)

def on_error(ws, error):
    print("### error ###")
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")
    print(f"Status code: {close_status_code}, message: {close_msg}")

def on_open(ws):
    def run(*args):
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_data = audio.raw_data
        
        frame_size = 8000  # 200ms
        intervel = 0.2  # 200ms
        
        status = 0  # 0: start, 1: continue, 2: end
        
        for i in range(0, len(audio_data), frame_size):
            if i + frame_size >= len(audio_data):
                status = 2
            else:
                status = 1 if i > 0 else 0
            
            frame_data = audio_data[i:i + frame_size]
            frame_base64 = base64.b64encode(frame_data).decode('utf-8')
            
            data = {
                "common": {"app_id": APP_ID},
                "business": {
                    "language": "zh_cn",
                    "domain": "iat",
                    "accent": "mandarin",
                    "vad_eos": 5000
                },
                "data": {
                    "status": status,
                    "format": "audio/L16;rate=16000",
                    "encoding": "raw",
                    "audio": frame_base64
                }
            }
            
            ws.send(json.dumps(data))
            time.sleep(intervel)
        
        time.sleep(1)
        ws.close()
    
    threading.Thread(target=run).start()

def transcribe_audio(audio):
    global audio_path, result_text
    result_text = ""
    audio_path = audio
    
    wsUrl = get_auth_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    return result_text

result_text = ""
audio_path = ""