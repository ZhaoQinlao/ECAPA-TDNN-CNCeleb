from flask import Flask,request,render_template
import demo_with_gradio
from pydub import AudioSegment
import io

app = Flask(__name__)

global embed_embeddings
embed_embeddings = []
global embed_file_names
embed_file_names = []
global user_names
user_names = []
global names
names = []
global record_files
record_files = []

def convert_to_wav(audio_file):
    # 将上传的文件读取为音频段
    audio_segment = AudioSegment.from_file(audio_file)
    # 转换为 WAV 格式
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="wav")
    # 返回包含 WAV 数据的 buffer
    buffer.seek(0)  # 重置 buffer 的位置到开始处
    return buffer

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        global record_files
        if record_files:
            embed_file_name, embed_file = record_files[0]
        else:
            embed_file = request.files['embed_file']
            embed_file_name = embed_file.filename
        global embed_file_names
        embed_file_names.append(embed_file_name)
        global embed_embeddings
        embed_embeddings.append(demo_with_gradio.get_embeddings(embed_file))
        global user_names
        user_names.append(request.form['username'])
        global names
        names = zip(user_names, embed_file_names)
    return render_template('index.html', names=names)

@app.route('/compare', methods=['POST'])
def compare():
    if request.method == 'POST':
        global record_files
        if record_files:
            eval_file_name, eval_file = record_files[0]
        else:
            eval_file = request.files['eval_file']
            eval_file_name=eval_file.filename
        eval_embedding = demo_with_gradio.get_embeddings(eval_file)
        #threshold = 0.54567164
        threshold = 0.1676453
        score=0.0
        scores=[]
        for embed_embedding in embed_embeddings:
            score = demo_with_gradio.detect_same_speaker_with_embeddings(embed_embedding, eval_embedding)
            scores.append(score)
        max_score = max(scores)
        max_speaker = user_names[scores.index(max_score)]
        if max_score > threshold:
            speaker = max_speaker
        else:
            speaker = "Unknown"
        global names
        names = list(zip(user_names, embed_file_names))
        return render_template('index.html', 
                               score=max_score, 
                               speaker=speaker, 
                               max_speaker=max_speaker, 
                               eval_file_name=eval_file_name, 
                               names=names)

@app.route('/records', methods=['POST'])
def records():
    if request.method == 'POST':
        record = request.files['audio']
        record_name = record.filename
        wav_buffer = convert_to_wav(record)
        global record_files
        record_files = []  # 清空列表
        record_files.append((record_name, wav_buffer))
        return '', 204

if __name__ == '__main__':
    app.run()