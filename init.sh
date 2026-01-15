#!/bin/bash

ENV_DIR=".STARGPT-env"
VENV_PY="$ENV_DIR/bin/python"

echo "🐍 Checking for python3.11..."
if ! command -v python3.11 &>/dev/null; then
    echo "🐍 python3.11 not found."

    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use Homebrew
        echo "🍎 macOS detected. Using Homebrew..."

        if ! command -v brew &>/dev/null; then
            echo "📦 Homebrew not found. Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

            # Add Homebrew to PATH for Apple Silicon
            if [[ $(uname -m) == "arm64" ]]; then
                echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
                eval "$(/opt/homebrew/bin/brew shellenv)"
            else
                echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zshrc
                eval "$(/usr/local/bin/brew shellenv)"
            fi
        fi

        echo "📦 Installing python@3.11 with Homebrew..."
        brew install python@3.11 || {
            echo "❌ Failed to install python@3.11"
            return 1
        }

        echo "🔗 Ensuring python3.11 is linked..."
        brew link python@3.11 --force || echo "⚠️ Could not link python3.11 (may already be linked)"

    else
        # Linux or other Unix - use pyenv (no sudo required)
        echo "🐧 Linux/Unix detected. Using pyenv (no sudo required)..."

        if ! command -v pyenv &>/dev/null; then
            echo "📦 pyenv not found. Installing pyenv..."
            curl https://pyenv.run | bash

            # Add pyenv to shell
            export PYENV_ROOT="$HOME/.pyenv"
            [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
            eval "$(pyenv init - bash)"

            # Add to bashrc if not already there
            if ! grep -q "PYENV_ROOT" ~/.bashrc; then
                echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
                echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
                echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
            fi
        else
            # Load pyenv
            export PYENV_ROOT="$HOME/.pyenv"
            [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
            eval "$(pyenv init - bash)"
        fi

        echo "📦 Installing Python 3.11.11 with pyenv..."
        pyenv install 3.11.11 || {
            echo "❌ Failed to install Python 3.11.11"
            return 1
        }

        echo "🔗 Setting Python 3.11.11 as global version..."
        pyenv global 3.11.11
    fi
else
    echo "🐍 python3.11 is installed."
fi

echo "🐍 Checking pip3.11..."
if ! command -v pip3.11 &>/dev/null; then
    echo "⚠️ pip3.11 missing. Trying ensurepip..."
    python3.11 -m ensurepip --upgrade || echo "⚠️ ensurepip failed; pip may already be installed."
else
    echo "🐍 pip3.11 is available."
fi

echo "🐍 Selecting pip3.11 as PIP..."
if command -v pip3.11 &>/dev/null; then
    PIP="$(command -v pip3.11)"
    echo "🐍 Using pip: $PIP"
else
    echo "❌ pip3.11 not found even though python3.11 exists"
    return 1
fi


echo "📦 Ensuring uv is installed in global Python..."
python3 -m pip install --upgrade pip setuptools wheel uv


# Create uv venv with Python 3.11 if it doesn't already exist
if [ ! -d "$ENV_DIR" ]; then
    echo "🌀 Creating uv venv ($ENV_DIR) with Python 3.11"
    python3 -m uv venv -p python3.11 "$ENV_DIR" || { echo "❌ Failed to create uv venv"; return 1; }
else
    echo "🌀 uv venv ($ENV_DIR) already exists. Skipping creation."
fi

# Activate the environment
if [ -f "$ENV_DIR/bin/activate" ]; then
    echo "🚀 Activating uv venv ($ENV_DIR)"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate" || echo "⚠️ Failed to activate venv; continuing..."
else
    echo "⚠️ Activation script not found in $ENV_DIR; did venv creation fail?"
fi

echo "🐍 Ensuring pip exists in the venv..."
$VENV_PY -m ensurepip --upgrade || { echo "❌ Failed to bootstrap pip in venv"; return 1; }

echo "📦 Ensuring uv is installed inside the venv..."
$VENV_PY -m pip install --upgrade uv || { echo "❌ Failed to install uv in venv"; return 1; }

# Install dependencies using uv
if [ -f "requirements.txt" ]; then
    # Increase timeout for slow/large package downloads (default is 30s)
    export UV_HTTP_TIMEOUT=120
    echo "📄 Installing dependencies from requirements.txt using uv"
    $VENV_PY -m uv pip install -r requirements.txt || { echo "❌ Dependency install failed"; return 1; }
else
    echo "⚠️ requirements.txt not found; skipping dependency install."
fi


echo "🛠 Ensuring repo root is on Python path via .pth"
# compute site-packages directory for this venv
SITE_PACKAGES=$("$ENV_DIR/bin/python" -c "import site; print(site.getsitepackages()[0])")
# write the repo root path into a .pth file
echo "$(pwd)" > "$SITE_PACKAGES/stargpt_paper.pth"


PROJECT_ROOT="$(pwd)"
SRC_PATH="$PROJECT_ROOT"
ACTIVATE="$ENV_DIR/bin/activate"
PY_PATH_LINE="export PYTHONPATH=\\"$PYTHONPATH:$SRC_PATH:$PROJECT_ROOT\\""

if ! grep -Fq "$PY_PATH_LINE" "$ACTIVATE"; then
    echo "🛠 Adding src/ and root to PYTHONPATH in venv activate script"
    echo "$PY_PATH_LINE" >> "$ACTIVATE"
else
    echo "🛠 Repo root already in PYTHONPATH for venv activate script"
fi

# Ensure .stargpt_midtrain directory exists
MIDTRAIN_DIR=".stargpt_midtrain"
if [ ! -d "$MIDTRAIN_DIR" ]; then
    echo "📁 Creating $MIDTRAIN_DIR directory"
    mkdir -p "$MIDTRAIN_DIR" || echo "⚠️ Failed to create $MIDTRAIN_DIR; continuing..."
else
    echo "📁 $MIDTRAIN_DIR already exists. Skipping creation."
fi

echo "🎉 Setup completed! To activate in a new shell, run: source $ENV_DIR/bin/activate"
