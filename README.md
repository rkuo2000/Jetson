# Jetson Orin Nano super

## Development Kit

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
### check OS version
`cat /etc/os-release`<br> 
```
PRETTY_NAME="Ubuntu 24.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.4 LTS (Noble Numbat)"
```

`uname -r`<br>
6.8.12-1021-tegra

`cat /etc/nv_tegra_release`<br>

`python -V`
```
Python 3.12.3
```

`sudo apt install python3-pip`<br>
`python3 -m pip install pip --upgrade`<br>

---
### venv setup
`sudo apt install python3-venv`<br>
`python3 -m venv .venv` <br>

---
### bash setup
**~/.bashrc** <br>
```
export PYTHONPATH=/usr/lib/python3.12/dist-packages:$PYTHONPATH
source ~/.venv/bin/activate
```

---
### install NodeJS
```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 26

node -v
npm -v
```

* `npm install -g npm@latest`

---
### OpenCode setup
```
npm install -g opencode-ai@latest
opencode -v
```

#### edit opencode.json
* Download opencode.json`
`cp ~/Downloads/opencode.json ~/.config/opencode`<br>
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
### [Gemma4 on Jetson](https://www.jetson-ai-lab.com/tutorials/gemma4-on-jetson/)
```
sudo docker run -it --rm --pull always --runtime=nvidia --network host -v $HOME/.cache/huggingface:/root/.cache/huggingface ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S
```
Model   Name : Gemma-4-E2B-it<br> 
Storage Size : ~3GB<br>


| 項      目 | 2B模型 | 3B模型 | 4GB以上 |
|-----------|-------|----------------|
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
### OpenCode operation
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
### [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)

#### install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```
cd ~/Open-LLM-VTuber
uv sync
uv run run_server.py
uv pip install edge-tts
edge-tts --version
```
* edit conf.yaml
```
llm_provider: 'vllm_llm'

```

```
uv run runserver.py
```
