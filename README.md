# [Jetson Orin Nano super](https://rkuo2000.github.io/Jetson/)

##  1. 開發者套件之介紹與安裝

### [NVIDIA Jetson Orin Nano Super 開發者套件](https://www.icshop.com.tw/products/368030502194)
<p>
<img width="25%" height="25%" src="https://shoplineimg.com/6486dbe2afaddb00694ea79f/6789f024c0286b000ff893dc/800x.webp?source_format=jpg">
<img width="25%" height="25%" src="https://shoplineimg.com/6486dbe2afaddb00694ea79f/6789f025a5e6f5000ee25aa5/800x.webp?source_format=jpg">
</p>

* DP轉HDMI 轉換線 4K 30Hz 15cm
* ADATA Legend 860 500GB PCIe 4.0 M.2 2280固態硬碟
<p>  
<img width="25%" height="25%" src="https://shoplineimg.com/6486dbe2afaddb00694ea79f/689452719e0c2a000e1bf878/800x.webp?source_format=jpg"> 
<img width="25%" height="25%" src="https://shoplineimg.com/6486dbe2afaddb00694ea79f/68901718d3174d00108c09b1/800x.webp?source_format=jpg">
</p>

---
### WebCam / BT Glasses
<p>
<img width="25%" src="https://github.com/rkuo2000/Jetson/blob/main/assets/JINPEI-webcam-1080p.webp?raw=true">
<img width="25%" src="https://github.com/rkuo2000/Jetson/blob/main/assets/Hyper_MZT.png?raw=true">
</p>

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
| NVIDIA TensorRT™ |	10.16.2 |
 
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
mkdir -p .cache/huggingface/hub
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

```
kokoro -t "Nice to meet you" -o output.mp3 --voice "af_heart"
```

---
### [Edge-TTS](https://github.com/rany2/edge-tts)
```
pipx install edge-tts
```

```
edge-tts --text "Hello, world!" --write-media output.mp3

edge-playback --text "Hello, world!
```

---
## 3. 本機語音辨識 (Local STT)

### [Whisper](https://github.com/openai/whisper/)
```
pip install git+https://github.com/openai/whisper.git 
```

```
whisper audio.flac audio.mp3 audio.wav --model turbo <br>

whisper japanese.wav --language Japanese <br>

whisper japanese.wav --model medium --language Japanese --task translate <br>
```

---
### [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)

#### [Export Whisper to ONNX](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/export-onnx.html)

#### [Sherpa-ONNX server](https://github.com/hfyydd/sherpa-onnx-server)
[server.js](https://github.com/hfyydd/sherpa-onnx-server/blob/main/server.js) - 支持多语言语音识别（中文、英文、日语、韩语、粤语）<br>

---
## 4. 本機語言模型 (Local LLM)

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
## 5. Agent 開發平台之安裝與操作

### [OpenCode setup](https://github.com/rkuo2000/AgenticCoding/blob/main/OpenCode.md)
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
## 6. Assistant 應用實作 

### Serve Gemma-4-E2B-it
```
sudo docker run -it --rm --pull always --runtime=nvidia --network host -v $HOME/.cache/huggingface:/root/.cache/huggingface ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
```

### [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)
*開源的LLM VTuber, 使用Google Gemma4-E2B-It模型, 含Agent, Web Search等技能*<br>
Download Release [Open-LLM-VTuber-v1.2.1-en.zip](https://pub-17317087be374bc68161ac63de2022a5.r2.dev/v1.2.1/Open-LLM-VTuber-v1.2.1-en.zip)<br>

#### install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 自動建立.venv 安裝python使用套件
```
cd ~/Open-LLM-VTuber
deactivate
uv sync
uv run run_server.py（有錯誤訊，按 ctrl-C結束）
uv pip install edge-tts
edge-tts --version
```

---
#### 設定使用模型 （本地 Gemma4-E2B-It)
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
#### 喇叭與麥克風設定
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
### [AI虛擬人（Live2D 語音助理）](https://github.com/YuriCrystal/ai-avatar-bot)
![](https://github.com/rkuo2000/Jetson/blob/main/assets/AI_Avatar.png?raw=true)

---
## 7. VLA 機器人實作 (Gemma4-VLA)
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

#### Build llama.cpp
```
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="87" \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
```

#### Download the model and vision projector
```
mkdir -p ~/models && cd ~/models

wget -O gemma-4-E2B-it-Q4_K_M.gguf https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf

wget -O mmproj-gemma4-e2b-f16.gguf https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/mmproj-gemma-4-E2B-it-bf16.gguf
```

#### Start the server
```
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj ~/models/mmproj-gemma4-e2b-f16.gguf \
  -c 2048 \
  --image-min-tokens 70 --image-max-tokens 70 \
  --ubatch-size 512 --batch-size 512 \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 --flash-attn on \
  --no-mmproj-offload --jinja -np 1
```
#### Verify server
```
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"Hi!"}],"max_tokens":32}' \
  | python3 -m json.tool
```

#### Find your mic, speaker, and webcam
**Mic**: `arecord -l` <br>
**Speaker**: `pactl list short sinks` <br>
**Webcam**: `v4l2-ctl --list-devices` <br>

#### Run the Demo
```
export MIC_DEVICE="plughw:0,0"
export SPK_DEVICE="alsa_output.platform-3510000.hda.HiFi__hw_HDA_3__sink"
export WEBCAM=0
export VOICE="af_jessica"
```
```
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

#### press space to record voice
```
Take a photo from webcam and analyze it
```

---
#### Gemma4_VLA Demo
![](https://github.com/rkuo2000/Jetson/blob/main/assets/Gemma4_VLA.png?raw=true)

#### llama.cpp server 
![](https://github.com/rkuo2000/Jetson/blob/main/assets/llama.cpp_server_gemma-4-E2B-it-Q4_K_M.png?raw=true)
![](https://github.com/rkuo2000/Jetson/blob/main/assets/llama.cpp_server_processing_image.png?raw=true)

---
### [GEM-4](https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/new-writeup-1778618527713)

#### Code: [https://github.com/takaki-maeda-99/GEM-4](https://github.com/takaki-maeda-99/GEM-4)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F33602339%2F6227624cde4a8fcec774ffb28add6a9f%2FGEM4.jpg?generation=1779067763846631&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F33602339%2F0ebdb094a641845b5f8c51aa28865610%2FVLA_archi.jpg?generation=1778807945930266&alt=media)

GEM-4: Gemma Embodied 4 Physical Assistance <br>
[![](https://markdown-videos-api.jorgenkh.no/youtube/OhaIA3bYwmg)](https://youtu.be/OhaIA3bYwmg)

---
## 8. [LeRobot 機器手臂](https://github.com/huggingface/lerobot)

### [SO-ARM101 AI 機器手臂PRO套件](https://www.icshop.com.tw/products/368040500233)

<iframe width="1038" height="584" src="https://www.youtube.com/embed/sD34HnAkGNc" title="Lerobot: An Open-Source Embodied Intelligence Algorithm" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

#### 商品規格:
| 型號         | SO-ARM101 Arm Kit Pro                                          |
|--------------|----------------------------------------------------------------|
| 主控手臂     | 1個 (7.4V) 1:345 齒輪比電機，用於 2 號關節                     |
|              | 2個（7.4V）1：191齒輪比電機，用於1號和3號關節                  |
|              | 3個（7.4V）1：145齒輪比電機，用於第4號、第5號關節和第6號夾持器 |
| 跟隨手臂     | 所有關節均配備 12 個（12V）1:345 齒輪比電機                    |
| 電源	       | 跟隨手臂：5.5mm*2.1mm DC 12V2A                                 |
|              | 主控手臂：5.5mm*2.1mm DC 5V4A                                  |
| 角度感測器   | 12位元磁編碼器                                                 |
| 工作溫度範圍 | 0℃ ～40℃                                                       |
| 溝通方式     | UART                                                           |
| 控制方法     | PC                                                             |

---
### [leRobot](https://huggingface.co/docs/lerobot/en/installation)
```
git clone https://github.com/huggingface/lerobot
cd lerobot
```
```
python -m venv .venv
source .venv/bin/activate
```

```
pip install -e ".[all]"
```
or 
```
pip install 'lerobot[all]'
```

#### Find Port & Setup Motors
```
lerobot-find-port

sudo chown usrname /dev/ttyACM2
lerobot-setup-motors --teleop.type=so101_leader --teleop.port=/dev/ttyACM0
```

```
lerobot-find-port

sudo chown usrname /dev/ttyACM3
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM3
```

---
#### Identify the Teleop ARM Port
```
export TELEOP_PORT=/dev/ttyACM2
export TELEOP_ID=my_leader_arm
```

#### Identify the Robot ARM Port
```
export ROBOT_PORT=/dev/ttyACM3
export ROBOT_ID=my_follower_arm
```

---
#### Calibration
<img width="50%" src="https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/_images/calibration_pose.jpg">

```
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=$TELEOP_PORT \
    --teleop.id=$TELEOP_ID
```

```
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=$ROBOT_PORT \
    --robot.id=$ROBOT_ID
```
The calibration file will then be saved in the ~/.cache/huggingface/lerobot/calibration directory<br>

![](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/_images/full_so101_calibration.gif)

---
#### Teleoperate
```
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=$ROBOT_PORT \
    --robot.id=$ROBOT_ID \
    --teleop.type=so101_leader \
    --teleop.port=$TELEOP_PORT \
    --teleop.id=$TELEOP_ID
```

#### Finding available Camera
```
lerobot-find-cameras opencv
```

#### Teleoperate with cameras
```
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=$ROBOT_PORT \
  --robot.id=$ROBOT_ID \
  --teleop.type=so101_leader \
  --teleop.port=$TELEOP_PORT \
  --teleop.id=$TELEOP_ID \
  --display_data=true \
  --robot.cameras='{
    "wrist": { "type": "opencv", "index_or_path": '"$CAMERA_GRIPPER"', "width": 640, "height": 480, "fps": 30, "rotation": "ROTATE_90_CLOCKWISE"},
    "front": { "type": "opencv", "index_or_path": '"$CAMERA_EXTERNAL"', "width": 640, "height": 480, "fps": 30 } }'
```

---
#### Record a dataset
```
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=$ROBOT_PORT \
    --robot.id=$ROBOT_ID \
    --robot.cameras="{ top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=$TELEOP_PORT \
    --teleop.id=$TELEOP_ID \
    --dataset.repo_id=${HF_USER}/so101_dataset_test \
    --dataset.num_episodes=30 \
    --dataset.single_task="put the red brick in a bowl" \
    --dataset.streaming_encoding=true \
    --display_data=true
```

---
#### LeLab (Web app)
LeLab is a web app that puts the full LeRobot workflow — calibrate, teleoperate, record, train, replay — into a single browser UI. Plug in your arm, open the app, and go. No CLI gymnastics, no keyboard prompts.<br>
```
uv tool install git+https://github.com/huggingface/leLab.git && lelab
```

---
### [Train Action Chunking Transformer (ACT) on SO-101](https://huggingface.co/blog/sherryxychen/train-act-on-so-101)
<img width="50%" src="https://cdn-uploads.huggingface.co/production/uploads/6885612c3bd4744a179e1f7f/PmyXORGA_nXPNfn5MVpYe.png">

---
### [Train SO101 Robot Sim-to-Real](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/05-building-workspace.html)
24GB VRAM required <br>

---
### [GR00T1.7 OpenVLA for LeRobot](https://huggingface.co/blog/nvidia/nvidia-isaac-teleop-and-gr00t17-in-lerobot)
![](https://cdn-uploads.huggingface.co/production/uploads/65563ac2b3bb6c3a41848a25/tArn-Nb0PJeQhMuGrHSiW.gif)
