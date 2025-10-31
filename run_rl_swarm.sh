#!/usr/bin/env bash

set -euo pipefail

# General arguments
ROOT=$PWD

# GenRL Swarm version to use
GENRL_TAG="0.1.11"

unset IDENTITY_PATH
export IDENTITY_PATH
export GENSYN_RESET_CONFIG
export CONNECT_TO_TESTNET=true
export ORG_ID
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_ENDPOINT="https://hf-mirror.com"
export SWARM_CONTRACT="0xFaD7C5e93f28257429569B854151A1B8DCD404c2"
export PRG_CONTRACT="0x51D4db531ae706a6eC732458825465058fA23a35"
export HUGGINGFACE_ACCESS_TOKEN="None"
export PRG_GAME=true

if [[ "$OSTYPE" == "darwin"* ]]; then
    # M4芯片专用内存优化设置
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    
    # 严格限制线程数（M4芯片优化）
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export NUMEXPR_MAX_THREADS=1
    export TOKENIZERS_PARALLELISM=false
    
    # MPS内存分配策略（16GB设备专用）
    export PYTORCH_MPS_ALLOCATOR_POLICY=delayed
    export PYTORCH_MPS_ALLOCATOR_POLICY_MAX_ALLOCATION=256
    export PYTORCH_MPS_MAX_MEMORY_FRACTION=0.25
    
    # Python内存优化
    export PYTHONHASHSEED=0
    export PYTHONDONTWRITEBYTECODE=1
    export MALLOC_ARENA_MAX=1
    
    # PyTorch CUDA配置（虽然是MPS，但某些设置仍有效）
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True
    
    # 强制小模型池（16GB设备）
    export FORCE_SMALL_MODEL=true
    
    # 启用梯度检查点以节省内存
    export ENABLE_GRADIENT_CHECKPOINTING=true
    
    # 限制Hugging Face缓存大小
    export HF_DATASETS_CACHE_MAX_SIZE=1000000000  # 1GB
    export TRANSFORMERS_CACHE_MAX_SIZE=2000000000  # 2GB
    
    # 使用国内镜像源
    export HF_ENDPOINT="https://hf-mirror.com"
    
    # 设置垃圾回收频率
    export PYTHONGC=1
fi

# Path to an RSA private key. If this path does not exist, a new key pair will be created.
# Remove this file if you want a new PeerID.
DEFAULT_IDENTITY_PATH="$ROOT"/swarm.pem
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}

DOCKER=${DOCKER:-""}
GENSYN_RESET_CONFIG=${GENSYN_RESET_CONFIG:-""}

# Bit of a workaround for the non-root docker container.
if [ -n "$DOCKER" ]; then
    volumes=(
        /home/gensyn/rl_swarm/modal-login/temp-data
        /home/gensyn/rl_swarm/keys
        /home/gensyn/rl_swarm/configs
        /home/gensyn/rl_swarm/logs
    )

    for volume in ${volumes[@]}; do
        sudo chown -R 1001:1001 $volume
    done
fi

# Will ignore any visible GPUs if set.
CPU_ONLY=${CPU_ONLY:-""}

# Set if successfully parsed from modal-login/temp-data/userData.json.
ORG_ID=${ORG_ID:-""}

GREEN_TEXT="\033[32m"
BLUE_TEXT="\033[34m"
RED_TEXT="\033[31m"
RESET_TEXT="\033[0m"

echo_green() {
    echo -e "$GREEN_TEXT$1$RESET_TEXT"
}

echo_blue() {
    echo -e "$BLUE_TEXT$1$RESET_TEXT"
}

echo_red() {
    echo -e "$RED_TEXT$1$RESET_TEXT"
}

ROOT_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"

# Function to clean up the server process upon final exit
final_cleanup() {
    echo_green ">> Shutting down trainer..."

    # Remove modal credentials if they exist
    rm -r $ROOT_DIR/modal-login/temp-data/*.json 2> /dev/null || true
    
    # Kill all processes belonging to this script's process group
    kill -- -$$ || true
}

errnotify() {
    echo_red ">> An error was detected while running rl-swarm. See $ROOT/logs for full logs."
}

# 设置最终清理，但不设置 EXIT trap，因为我们需要重试逻辑工作
trap final_cleanup SIGINT SIGTERM
trap errnotify ERR

echo -e "\033[38;5;224m"
cat << "EOF"
    ██████  ██            ███████ ██     ██  █████  ██████  ███    ███
    ██   ██ ██            ██      ██     ██ ██   ██ ██   ██ ████  ████
    ██████  ██      █████ ███████ ██  █  ██ ███████ ██████  ██ ████ ██
    ██   ██ ██                 ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██
    ██   ██ ███████       ███████  ███ ███  ██   ██ ██   ██ ██      ██

    From Gensyn

EOF

# Create logs directory if it doesn't exist
mkdir -p "$ROOT/logs"

if [ "$CONNECT_TO_TESTNET" = true ]; then
    # Run modal_login server.
    echo "Please login to create an Ethereum Server Wallet"
    cd modal-login
    # Check if the yarn command exists; if not, install Yarn.

    # Node.js + NVM setup
    if ! command -v node > /dev/null 2>&1; then
        echo "Node.js not found. Installing NVM and latest Node.js..."
        export NVM_DIR="$HOME/.nvm"
        if [ ! -d "$NVM_DIR" ]; then
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
        fi
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
        nvm install node
    else
        echo "Node.js is already installed: $(node -v)"
    fi

    if ! command -v yarn > /dev/null 2>&1; then
        # Detect Ubuntu (including WSL Ubuntu) and install Yarn accordingly
        if grep -qi "ubuntu" /etc/os-release 2> /dev/null || uname -r | grep -qi "microsoft"; then
            echo "Detected Ubuntu or WSL Ubuntu. Installing Yarn via apt..."
            curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
            echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
            sudo apt update && sudo apt install -y yarn
        else
            echo "Yarn not found. Installing Yarn globally with npm (no profile edits)…"
            # This lands in $NVM_DIR/versions/node/<ver>/bin which is already on PATH
            npm install -g --silent yarn
        fi
    fi

    ENV_FILE="$ROOT"/modal-login/.env
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS version
        sed -i '' "3s/.*/SWARM_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
        sed -i '' "4s/.*/PRG_CONTRACT_ADDRESS=$PRG_CONTRACT/" "$ENV_FILE"

    else
        # Linux version
        sed -i "3s/.*/SWARM_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
        sed -i "4s/.*/PRG_CONTRACT_ADDRESS=$PRG_CONTRACT/" "$ENV_FILE"
    fi


    # Docker image already builds it, no need to again.
    if [ -z "$DOCKER" ]; then
        yarn install --immutable
        echo "Building server"
        yarn build > "$ROOT/logs/yarn.log" 2>&1
    fi
    yarn start >> "$ROOT/logs/yarn.log" 2>&1 & # Run in background and log output

    SERVER_PID=$!  # Store the process ID
    echo "Started server process: $SERVER_PID"
    sleep 5

    cd ..

    echo_green ">> Waiting for modal userData.json to be created..."
    while [ ! -f "modal-login/temp-data/userData.json" ]; do
        sleep 5  # Wait for 5 seconds before checking again
    done
    echo "Found userData.json. Proceeding..."

    ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' modal-login/temp-data/userData.json)
    echo "Your ORG_ID is set to: $ORG_ID"

    # Wait until the API key is activated by the client
    echo "Waiting for API key to become activated..."
    while true; do
        STATUS=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID")
        if [[ "$STATUS" == "activated" ]]; then
            echo "API key is activated! Proceeding..."
            break
        else
            echo "Waiting for API key to be activated..."
            sleep 5
        fi
    done
fi

echo_green ">> Getting requirements..."
pip install --upgrade pip

 echo_green ">> Installing GenRL..."
pip install gensyn-genrl==${GENRL_TAG}
pip install reasoning-gym>=0.1.20 # for reasoning gym env
pip install hivemind@git+https://github.com/gensyn-ai/hivemind@639c964a8019de63135a2594663b5bec8e5356dd # We need the latest, 1.1.11 is broken


if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' -E 's/(startup_timeout: *float *= *)[0-9.]+/\1120/' $(python3 -c "import hivemind.p2p.p2p_daemon as m; print(m.__file__)")
else
    sed -i -E 's/(startup_timeout: *float *= *)[0-9.]+/\1120/' $(python3 -c "import hivemind.p2p.p2p_daemon as m; print(m.__file__)")
fi

if [ ! -d "$ROOT/configs" ]; then
    mkdir "$ROOT/configs"
fi  

if [ -f "$ROOT/configs/rg-swarm.yaml" ]; then
    echo_green ">> Found differences in rg-swarm.yaml. Backing up existing config."
    rm "$ROOT/configs/rg-swarm.yaml"
    cp "$ROOT/rgym_exp/config/rg-swarm.yaml" "$ROOT/configs/rg-swarm.yaml"
else
    cp "$ROOT/rgym_exp/config/rg-swarm.yaml" "$ROOT/configs/rg-swarm.yaml"
fi

if [ -n "$DOCKER" ]; then
    # Make it easier to edit the configs on Linux systems.
    sudo chmod -R 0777 /home/gensyn/rl_swarm/configs
fi

echo_green ">> Done!"

HF_TOKEN=${HF_TOKEN:-""}
HUGGINGFACE_ACCESS_TOKEN="None"

# echo -en $GREEN_TEXT
# read -p ">> Would you like to push models you train in the RL swarm to the Hugging Face Hub? [y/N] " yn
# echo -en $RESET_TEXT
# yn=${yn:-N} # Default to "N" if the user presses Enter
# case $yn in
#     [Yy]*) read -p "Enter your Hugging Face access token: " HUGGINGFACE_ACCESS_TOKEN ;;
#     [Nn]*) HUGGINGFACE_ACCESS_TOKEN="None" ;;
#     *) echo ">>> No answer was given, so NO models will be pushed to Hugging Face Hub" && HUGGINGFACE_ACCESS_TOKEN="None" ;;
# esac


# echo -en $GREEN_TEXT
# read -p ">> Enter the name of the model you want to use in huggingface repo/name format, or press [Enter] to use the default model. " MODEL_NAME
# echo -en $RESET_TEXT

# Only export MODEL_NAME if user provided a non-empty value
# if [ -n "$MODEL_NAME" ]; then
#     export MODEL_NAME
#     echo_green ">> Using model: $MODEL_NAME"
# else
#     echo_green ">> Using default model from config"
# fi

# echo -en $GREEN_TEXT
# read -p ">> Would you like your model to participate in the AI Prediction Market? [Y/n] " yn
# if [ "$yn" = "n" ] || [ "$yn" = "N" ]; then
#     PRG_GAME=false
#     echo_green ">> Playing PRG game: false"
# else
#     echo_green ">> Playing PRG game: true"
# fi

if [ "$PRG_GAME" = "true" ]; then 
    echo_green ">> Playing PRG game: true"
else
    echo_green ">> Playing PRG game: false"
fi

echo -en $RESET_TEXT
echo_green ">> Good luck in the swarm!"
echo_blue ">> And remember to star the repo on GitHub! --> https://github.com/gensyn-ai/rl-swarm"

cleanup_for_retry() {
    echo_green ">> Cleaning up before retry..."
    
    # 清理可能残留的python进程
    pkill -f "rgym_exp.runner.swarm_launcher" 2>/dev/null || true
    pkill -f "python.*rgym_exp" 2>/dev/null || true
    
    # 检查modal-login服务器状态
    if [ "$CONNECT_TO_TESTNET" = true ]; then
        if lsof -i :3000 >/dev/null 2>&1; then
            echo_green ">> Modal-login server is running on port 3000"
        else
            echo_red ">> Modal-login server is not running. Restarting..."
            
            # 杀死可能残留的yarn/node进程
            pkill -f "modal-login" 2>/dev/null || true
            pkill -f "yarn.*start" 2>/dev/null || true
            sleep 2
            
            cd modal-login
            yarn start >> "$ROOT/logs/yarn.log" 2>&1 &
            SERVER_PID=$!
            echo ">> Restarted modal-login server with PID: $SERVER_PID"
        fi
    fi
    
    # 清理可能的锁文件或临时文件
    rm -f "$ROOT"/*.lock 2>/dev/null || true
    rm -f "$ROOT"/logs/*.lock 2>/dev/null || true
    
    # 清理旧的训练日志文件（保留最近的5个）
    if [ -d "$ROOT/logs" ]; then
        find "$ROOT/logs" -name "training_*.log" -type f | sort -r | tail -n +6 | xargs rm -f 2>/dev/null || true
    fi
    
    # 等待一下让系统资源释放
    sleep 2
}

# 启动训练的函数
start_training() {
    echo_green ">> Starting RL Swarm training..."
    
    # 创建训练日志文件
    local training_log="$ROOT/logs/training_$(date +%Y%m%d_%H%M%S).log"
    
    # 启动训练进程到后台并获取PID
    python -m rgym_exp.runner.swarm_launcher \
        --config-path "$ROOT/rgym_exp/config" \
        --config-name "rg-swarm.yaml" > "$training_log" 2>&1 &
    
    local training_pid=$!
    echo_blue ">> Training process started with PID: $training_pid"
    echo_blue ">> Training log: $training_log"
    
    # 启动日志实时显示
    tail -f "$training_log" &
    local tail_pid=$!
    
    # 等待训练进程完成
    wait $training_pid
    local exit_code=$?
    
    # 停止日志显示
    kill $tail_pid 2>/dev/null || true
    
    echo_blue ">> Training process completed with exit code: $exit_code"
    
    # 检查训练日志中是否有DHT连接错误或其他严重错误
    if [ -f "$training_log" ]; then
        # 检查是否有典型的错误模式
        if grep -q "TypeError: cannot unpack non-iterable NoneType object" "$training_log" 2>/dev/null; then
            echo_red ">> Detected DHT connection error in training logs"
            echo_red ">> Error details saved to: $training_log"
            return 1  # 强制返回错误码
        fi
        
        if grep -q "Exception occurred during game run" "$training_log" 2>/dev/null; then
            echo_red ">> Detected game runtime exception in training logs"
            echo_red ">> Error details saved to: $training_log"
            return 1  # 强制返回错误码
        fi
        
        # 检查其他常见的错误模式
        if grep -q "Traceback (most recent call last)" "$training_log" 2>/dev/null; then
            echo_red ">> Detected Python exception in training logs"
            echo_red ">> Error details saved to: $training_log"
            return 1  # 强制返回错误码
        fi
    fi
    
    return $exit_code
}

# 带重试逻辑的训练函数
run_training_with_retry() {
    local MAX_RETRIES=60
    local RETRY_DELAY=30
    
    for attempt in $(seq 1 $MAX_RETRIES); do
        echo_green ">> Training attempt $attempt/$MAX_RETRIES"
        
        set +e  # 手动捕获退出码，避免 -e 导致提前退出
        start_training
        exit_code=$?
        set -e
        
        if [ $exit_code -eq 0 ]; then
            echo_green ">> Training completed successfully!"
            final_cleanup
            exit 0
        elif [ $exit_code -eq 137 ]; then
            echo_red ">> Training attempt $attempt was killed by system (SIGKILL)"
            echo_red ">> This usually indicates memory issues or system resource constraints"
        elif [ $exit_code -eq 143 ]; then
            echo_red ">> Training attempt $attempt was terminated (SIGTERM or timeout)"
            echo_red ">> This may indicate the process was stuck or killed due to inactivity"
         elif [ $exit_code -eq 1 ]; then
            # DHT连接失败或其他运行时错误
            echo_red ">> Training attempt $attempt failed with DHT/runtime error (exit code 1)"
            echo_red ">> This usually indicates network connectivity or DHT synchronization issues"
        else
            echo_red ">> Training attempt $attempt failed with exit code $exit_code"
        fi
        
        if [ $attempt -lt $MAX_RETRIES ]; then
            cleanup_for_retry
            echo_green ">> Waiting ${RETRY_DELAY}s before retry..."
            sleep $RETRY_DELAY
        else
            echo_red ">> All $MAX_RETRIES attempts failed. Final exit code: $exit_code"
            final_cleanup
            exit 1
        fi
    done
}


# 现在开始重试逻辑
run_training_with_retry
