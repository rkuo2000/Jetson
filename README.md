# Jetson Orin Nano super

##  1. AI Agent 開發者套件之介紹與安裝

### [NVIDIA Jetson Orin Nano Super 開發者套件](https://www.icshop.com.tw/products/368030502194)
<p>
<img width="25%" height="25%" src="https://shoplineimg.com/6486dbe2afaddb00694ea79f/6789f024c0286b000ff893dc/800x.webp?source_format=jpg">
<img width="25%" height="25%" src="https://shoplineimg.com/6486dbe2afaddb00694ea79f/6789f025a5e6f5000ee25aa5/800x.webp?source_format=jpg">
</p>

* DP轉HDMI 轉換線 4K 30Hz 15cm
* ADATA Legend 860 500GB PCIe 4.0 M.2 2280固態硬碟
<p>  
<img width="20%" height="20%" src="https://shoplineimg.com/6486dbe2afaddb00694ea79f/689452719e0c2a000e1bf878/800x.webp?source_format=jpg"> 
<img width="20%" height="20%" src="https://shoplineimg.com/6486dbe2afaddb00694ea79f/68901718d3174d00108c09b1/800x.webp?source_format=jpg">
</p>

---
### [藍牙眼鏡 (Hyper MZT)](https://shopee.tw/%E2%9C%A8%E5%90%8C%E8%81%B2%E5%82%B3%E8%AD%AF%E2%9C%A8AI%E7%BF%BB%E8%AD%AF%E7%9C%BC%E9%8F%A1-AI%E8%81%BD%E6%AD%8C%E7%9C%BC%E9%8F%A1-AI%E6%8B%8D%E7%85%A7%E7%9C%BC%E9%8F%A1-%E6%99%BA%E8%83%BDAI%E7%9C%BC%E9%8F%A1-%E7%BF%BB%E8%AD%AF%E7%9C%BC%E9%8F%A1-100-%E8%AA%9E%E8%A8%80%E7%BF%BB%E8%AD%AF-%E5%8F%AF%E6%9C%83%E8%AD%B0%E5%AF%A6%E6%99%82%E8%A8%98%E9%8C%84-%E5%AF%A6%E6%99%82%E7%BF%BB%E8%AD%AF-i.1684529157.51060102277)
![](https://github.com/rkuo2000/Jetson/blob/main/assets/Hyper_MZT.png?raw=true)

---
### [使用NVIDIA SDK Manager安裝系統至SSD開機](https://blog.cavedu.com/2025/02/14/nvidia-jetson-orin-nano-super/)
* 用Ubuntu PC由網址選擇deb檔案安裝 [https://developer.nvidia.com/sdk-manager](https://developer.nvidia.com/sdk-manager)
* 要使用NVIDIA SDK Manager 來燒錄Jetson Orin Nano作業系統需要先將Jetson Orin Nano進入Recovery mode進行手動安裝安裝。
* 讓板子進入Recovery Mode的做法是用 jumper 插上pin9 與 pin10(FC REC,GND)，之後再通電。
#### [Jetpack 7.2](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2)
| Features | Versions |
|----------|----------|
| Linux  | R39.2 |
| Kernel | K6.8 |
| Distro | L4T Ubuntu 24.04 |
| CUDA   | 13.2.1 |
| NVIDIA CuDNN 	| 9.20.0 |
| NVIDIA TensorRT™ |	10.16.2
 
---
### 作業系統之環境工具設定
`cat /etc/os-release`<br> 
```
PRETTY_NAME="Ubuntu 24.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.4 LTS (Noble Numbat)"
```

`uname -r`<br>
6.8.12-1021-tegra <br>

`cat /etc/nv_tegra_release`<br>

`python -V`
```
Python 3.12.3
```

`sudo apt install python3-pip`<br>
`python3 -m pip install pip --upgrade`<br>

---
#### venv setup
```
sudo apt install python3-venv
python3 -m venv .venv
```

---
#### bash setup
**~/.bashrc** <br>
```
export PYTHONPATH=/usr/lib/python3.12/dist-packages:$PYTHONPATH
source ~/.venv/bin/activate
```

---
#### install NodeJS
```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 26

node -v
npm -v
```

* `npm install -g npm@latest`

---
## 2. 本機語音合成 (Local TTS)

### [Kokoro](https://github.com/hexgrad/kokoro)
```
pip install kokoro-tts soundfile
apt install espeak-ng
```

kokoro -t "Nice to meet you" -o output.mp3 --voice "al_heart" <br>

---
### [Edge-TTS](https://github.com/rany2/edge-tts)
```
pipx install edge-tts
```

edge-tts --text "Hello, world!" --write-media output.mp3 <br>

edge-playback --text "Hello, world! <br>

---
## 3. 本機語音辨識 (Local ASR)

### [Whisper](https://github.com/openai/whisper/)
```
pip install git+https://github.com/openai/whisper.git 
```

whisper audio.flac audio.mp3 audio.wav --model turbo <br>

whisper japanese.wav --language Japanese <br>

whisper japanese.wav --model medium --language Japanese --task translate <br>

---
### [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)

#### [Export Whisper to ONNX](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/export-onnx.html)

#### [Sherpa-ONNX server](https://github.com/hfyydd/sherpa-onnx-server)
[server.js](https://github.com/hfyydd/sherpa-onnx-server/blob/main/server.js) - 支持多语言语音识别（中文、英文、日语、韩语、粤语）<br>

---
## 4. 本機語言模型：Gemma4-E2B

### Model : [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
Storage Size : **3GB** <br>

| 項      目 | 2B模型 | 3B模型 | 4GB以上 |
|-----------|-------|--------|--------|
| Property  |	E2B | 	E4B |	31B Dense |
|Total Parameters | 	2.3B effective (5.1B with embeddings) |	4.5B effective (8B with embeddings) |	30.7B
| Layers |	35 |	42 |	60 |
| Sliding Window |	512 tokens |	512 tokens | 	1024 tokens
| Context Length |	128K tokensi| x	128K tokens |	256K tokens |
| Vocabulary Size |	262K |	262K |	262K |
| Supported Modalitiesi | 	Text, Image, Audio |	Text, Image, Audio |	Text, Image |
| Vision Encoder Parameters |	~150M | ~150M | ~550M    |
| Audio Encoder Parametersi |	~300M | ~300M |	No Audio |

---
### [Gemma4 on Jetson](https://www.jetson-ai-lab.com/tutorials/gemma4-on-jetson/)
```
sudo docker run -it --rm --pull always --runtime=nvidia --network host -v $HOME/.cache/huggingface:/root/.cache/huggingface ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S
```

---
## 5. AI Agent 開發平台之安裝與操作

### OpenCode setup
```
npm install -g opencode-ai@latest
opencode -v
```

#### edit opencode.json
* Download opencode.json

`cp ~/Downloads/opencode.json ~/.config/opencode` <br>

```
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama_cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama_cpp (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "Gemma-4-E2B-It": {
          "name": "Gemma-4-E2B-It",
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "tools": true,
          "reasoning": true
        }
      }
    }
  },
  "model": "llama_cpp/gemma4:e2b",
  "mcp": {
    "ameba-pro2": {
      "type": "local",
      "command": [
        "uv",
        "--directory",
        "/home/rkuo/ameba-mcp/src/ameba-mcp-server",
        "run",
        "ameba-mcp",
        "--product",
        "ameba-pro2"
      ]
    }
  }
}
```

---
### OpenCode 操作與測試
```
git clone https://github.com/rkuo2000/AgenticCoding
cd ~/AgenticCoding/
opencode
```
or
```
opencode web
```

---
## 6. AI Agent 應用實作 ~ [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)
*開源的LLM VTuber, 使用Google Gemma4-E2B-It模型, 含Agent, Web Search等技能*<br>

### install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 自動建立.venv 安裝python使用套件
```
cd ~/Open-LLM-VTuber
deactivate
uv sync
uv run run_server.py（有錯誤訊，按 ctrl-C結束）
uv pip install edge-tts
edge-tts --version
```

---
### 設定使用模型 （本地 Gemma4-E2B-It)
`vi conf.yaml`<br>
```
     ollama_llm:
        base_url: 'http://localhost:8080/v1'
        model: 'Gemma4-E2B-It'
        temperature: 1.0 # value between 0 to 2
        # seconds to keep the model in memory after inactivity. 
        # set to -1 to keep the model in memory forever (even after exiting open llm vtuber)
        keep_alive: -1
        iunload_at_exit: True # unload the model from memory at exit
```
`python run_server.py`<br>

---
### 喇叭與麥克風設定
**Ubuntu - Settings > Sound**<br>
* Input : `WebCam C310`<br>
* Output: `HDMI Display port - Built-in Audio`<br>
![](https://github.com/rkuo2000/Jetson/blob/main/assets/Ubuntu_Settings_Sound.png?raw=true)

---
### 聊天機器人 Demo
![](https://github.com/rkuo2000/Jetson/blob/main/assets/Open_LLM_VTuber.png?raw=true)
**自強基金會簡介** : `https://edu.tcfst.org.tw/web/tw/about/index.asp`<br>
**國立台灣海洋大學電機工程系簡介** : `https://ee.ntou.edu.tw/p/412-1062-7466.php?Lang=zh-tw`<br>

---
### [Hermes + JARVIS（鋼鐵人助理）](https://github.com/eadmin2/jarvis_ai)
![](https://github.com/rkuo2000/Jetson/blob/main/assets/Hermes_JARVIS.png?raw=true)

---
### [AI虛擬人（Live2D 語音助理）](https://github.com/YuriCrystal/ai-avatar-bot)
![](https://github.com/rkuo2000/Jetson/blob/main/assets/AI_Avatar.png?raw=true)

---
## 7. VLA 機器人 (Jetson Gemma4)
### [Gemma 4 VLA Demo on Jetson Orin Nano Super](https://huggingface.co/blog/nvidia/gemma4)
```
You speak → Parakeet STT → Gemma 4 → [Webcam if needed] → Kokoro TTS → Speaker
```
#### [Code](https://github.com/asierarranz/Google_Gemma.git)
```
wget https://raw.githubusercontent.com/asierarranz/Google_Gemma/main/Gemma4/Gemma4_vla.py
```

#### Harewares
* NVIDIA Jetson Orin Nano Super (8 GB)
* Logitech C920 webcam (mic built in)
* USB speaker
* USB keyboard (to press SPACE)

```
pip install opencv-python-headless onnx_asr kokoro-onnx soundfile huggingface-hub numpy
```

#### Add some swap
```
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```
#### Kill memory hogs
```
sudo systemctl stop docker 2>/dev/null || true
sudo systemctl stop containerd 2>/dev/null || true
pkill -f tracker-miner-fs-3 || true
pkill -f gnome-software || true
free -h
```

#### Serve Gemma4

#### Find your mic, speaker, and webcam
**Mic**: `arecord -l` <br>
**Speaker**: `pactl list short sinks` <br>
**Webcam**: `v4l2-ctl --list-devices` <br>

#### Run the Demo
```
source .venv/bin/activate

export MIC_DEVICE="plughw:3,0"
export SPK_DEVICE="alsa_output.usb-Generic_USB2.0_Device_20130100ph0-00.analog-stereo"
export WEBCAM=0
export VOICE="af_jessica"

python3 Gemma4_vla.py
```
* Text mode:<br>
```
python3 Gemma4_vla.py --text
```
* Change Voice<br>
```
export VOICE="am_puck"
python3 gemma4_vla.py
```

#### The script exposes exactly one tool to Gemma 4:
```
{
  "name": "look_and_answer",
  "description": "Take a photo with the webcam and analyze what is visible."
}
```
