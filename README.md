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

---
### check OS version
* `cat /etc/os-release` 
```
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
```

* `uname -r`
5.15.148-tegra

* `cat /etc/nv_tegra_release`
```
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidi
```

* `python -V`
```
Python 3.10.12
```

---
### venv setup
`python3 -m pip install pip` <br>
`python3 -m venv .yolo` <br>

---
### bash setup
**~/.bashrc** <br>
```
export PYTHONPATH=/usr/lib/python3.12/dist-packages:$PYTHONPATH
source ~/.yolo/bin/activate  # commented out by conda initialize
cd ~
```
---
### node & npm install
```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
nvm install node
```

* `node -v`
v25.6.1

* `npm -v`
v 11.9.0

* `npm install -g npm@latest`
v 11.10.0

---
### OpenClaw setup
[![](https://markdown-videos-api.jorgenkh.no/youtube/daXOXSSyudM)](https://youtu.be/daXOXSSyudM)

#### install [OpenClaw](https://github.com/openclaw/openclaw)
1. `sudo npm install -g openclaw@latest`
2. `openclaw -v`
3. `openclaw onboard --install-daemon`
4. `openclaw gateway restart`
5. open browser `http://127.0.0.1:18789`

[.openclaw/openclaw.json](https://github.com/rkuo2000/GenAI/blob/main/Agent/openclaw.json)<br>

---
#### setup Ollama
add the following into `~/.openclaw/openclaw.json` <br>

```
  "models": {
    "mode": "merge",
    "providers": {
      "ollama": {
        "baseUrl": "http://192.168.0.13:11434/v1",
        "apiKey": "ollama",
        "api": "openai-responses",
        "models": [
          {
            "id": "gpt-oss:latest",
            "name": "GPT-OSS:20b (Local)",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 32768,
            "maxTokens": 4096
          }
        ]
      }
    }
```
To access a remote Ollama server: <br>
* modify openclaw.json, *replace `127.0.0.1` to `192.168.0.12` (remote ip addr)* 
* modify ufw rules on Ollama server, *`sudo ufw allow from 192.168.0.18`*
* 
---
#### setup WhatsApp
*.openclaw/openclaw.json*<br>
```
  "channels": {
    "whatsapp": {
      "selfChatMode": true,
      "dmPolicy": "allowlist",
      "allowFrom": [
        "+886972123456"
      ]
    }
  },
```

---
#### setup Gmail
* **API和服務**
  - **建立專案** [Google Console && create project](https://console.cloud.google.com/projectcreate)
  - **專案名稱** `Openclaw-Gmail-API`
* **API和服務** ==> **+啟用API和服務** ==> **[Gmail API]** ==> Enable
* **憑證** ==> **建立憑證** ==> **OAuth用戶端ID**
  - **應用程式類型** : 選`電腦版應用程式`
  - **名稱** : 填`OpenClaw` ==> 按`建立` ==> 下載JSON
  - 下載後改名 `client_secret.json` 移至`.openclaw/workspace`
* 在`localhost:18789`, prompt輸入 `read .openclaw/workspace/client_secret.json and make a gmail-auth.py to access Gmail API`
* 自動會在workspace中產生 gmail_auth.py
* `pip install --upgrade google-auth-oauthlib google-auth-httplib2`
* `python gmail-auth.py`
* 執行後會開啟瀏覽器，選定Gmail帳號，按**繼續** 即可完成授權。
  
---
#### setup VPN : Tailscale
```
curl -fsSL <https://tailscale.com/install.sh> | sh
sudo tailscale up
```

#### setup Firewall
```
sudo apt install ufw -y
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow in on tailscale0 to any port 22
sudo ufw enable #Type 『y』 to confirm`
sudo ufw status
```
