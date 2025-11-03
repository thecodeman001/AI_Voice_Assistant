# AI Voice Assistant - Bank Support Training

> Real-time voice simulation for bank customer support training with ASR→LLM→TTS pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Groq](https://img.shields.io/badge/Groq-ASR%20%26%20LLM-orange.svg)](https://groq.com/)
[![Kokoro](https://img.shields.io/badge/Kokoro-TTS-purple.svg)](https://github.com/hexgrad/Kokoro-82M)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [System Requirements](#system-requirements)
- [Complete Installation Guide](#complete-installation-guide)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [License](#license)

---

## Overview

An AI-powered voice assistant designed for bank customer support training. The system provides realistic voice interactions using state-of-the-art technologies:

- **ASR (Speech Recognition)**: Groq Whisper (whisper-large-v3-turbo)
- **LLM (Language Model)**: Groq LLM for intelligent responses
- **TTS (Text-to-Speech)**: Kokoro-82M for natural voice (local, offline)

### Use Cases
- Bank support agent training and assessment
- Customer service simulation scenarios
- Voice interface prototyping and testing
- Multi-modal AI demonstrations

---

## Features

### Core Capabilities
- **Manual Recording Controls** - Explicit START/STOP buttons for precise control
- **High-Accuracy ASR** - Groq Whisper for speech transcription
- **Intelligent Responses** - Scenario-specific AI behavior and prompts
- **Natural Voice** - Kokoro-82M local TTS (11+ voices, runs offline)
- **Three Scenarios** - Lost Card, Failed Transfer, Locked Account
- **Performance Metrics** - Real-time latency tracking (ASR, LLM, TTS)
- **Conversation History** - Multi-turn dialogue with context
- **State Management** - Context-aware conversation flow

### Technical Highlights
- **Local TTS** - Zero API costs for voice synthesis, runs offline
- **Streaming Pipeline** - Real-time audio processing
- **Modular Architecture** - Easy to extend and maintain
- **Error Handling** - Comprehensive exception management
- **Production Ready** - Clean code, proper logging, tested

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Streamlit Web UI                    │
│       [START RECORDING] [STOP & PROCESS]         │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│          SimpleVoiceHandler Pipeline             │
│                                                  │
│  Record → ASR → LLM → TTS → Playback            │
│    ↓       ↓     ↓     ↓       ↓                │
│  Audio   Text  Reply Audio  Speaker             │
└────┬──────┬──────┬──────┬────────────────────────┘
     │      │      │      │
     ▼      ▼      ▼      ▼
  Mic    Groq   Groq  Kokoro
        Whisper  LLM   TTS
                      (Local)
```

**Pipeline Flow:**
1. **User speaks** → Microphone captures audio
2. **ASR** → Groq Whisper transcribes to text
3. **LLM** → Groq generates intelligent response
4. **TTS** → Kokoro synthesizes speech (offline)
5. **Playback** → User hears AI response

---

## System Requirements

### Hardware
- **CPU**: Modern multi-core processor (Intel/AMD/Apple Silicon)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: ~500MB for dependencies and models
- **Microphone**: Any working microphone (built-in or external)
- **Speakers/Headphones**: For audio output

### Software
- **Operating System**: macOS, Linux, or Windows
- **Python**: Version 3.11 or higher
- **Internet**: Required for ASR/LLM API calls (Groq)
  - TTS works offline (Kokoro is local)

---

## Complete Installation Guide

Follow these steps carefully to set up the system on your local machine.

### Step 1: Install Python

**Check if Python is installed:**
```bash
python --version
# or
python3 --version
```

You need Python 3.11 or higher. If not installed:

**macOS:**
```bash
# Using Homebrew (install Homebrew first if needed: https://brew.sh)
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer
3. **IMPORTANT**: Check "Add Python to PATH" during installation

**Verify installation:**
```bash
python3 --version
# Should show: Python 3.11.x or higher
```

---

### Step 2: Install espeak-ng (Required for TTS)

The Kokoro TTS engine requires espeak-ng for phoneme generation.

**macOS:**
```bash
brew install espeak-ng
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install espeak-ng
```

**Windows:**
1. Download installer from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)
2. Run the installer (choose default options)
3. Add installation directory to PATH:
   - Default: `C:\Program Files\eSpeak NG\`
   - Add to System PATH in Environment Variables

**Verify installation:**
```bash
espeak-ng --version
# Should show: eSpeak NG version info
```

---

### Step 3: Get Groq API Key

You need a free Groq API key for ASR and LLM.

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account (no credit card required)
3. Navigate to **API Keys** section
4. Click **"Create API Key"**
5. **Copy and save the key** (you'll need it in Step 7)

**Example key format:** `gsk_...` (starts with gsk_)

---

### Step 4: Download the Project

**Option A: Using Git**
```bash
git clone <your-repository-url>
cd Voice_Test_Project
```

**Option B: Download ZIP**
1. Download the project ZIP file
2. Extract to your desired location
3. Open terminal/command prompt and navigate:
   ```bash
   cd /path/to/Voice_Test_Project
   ```

---

### Step 5: Create Virtual Environment

A virtual environment keeps project dependencies isolated from your system Python.

**Create the environment:**
```bash
python3 -m venv venv
```

**Activate the environment:**

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Success indicator:** You should see `(venv)` at the beginning of your terminal prompt.

Example:
```
(venv) user@computer Voice_Test_Project %
```

---

### Step 6: Install Python Dependencies

**Upgrade pip first:**
```bash
pip install --upgrade pip
```

**Install all required packages:**
```bash
pip install -r requirements.txt
```

This installs 100+ packages including:
- `streamlit` - Web UI framework
- `groq` - ASR and LLM API client
- `kokoro` - Local TTS engine
- `sounddevice` - Audio recording/playback
- `numpy`, `torch` - Audio processing
- `loguru` - Logging
- And many dependencies

**⏱️ Installation takes 3-5 minutes. Wait for completion.**

**Verify installation:**
```bash
python -c "import streamlit, groq, kokoro, sounddevice; print('✅ All core modules installed!')"
```

---

### Step 7: Configure Environment Variables

**Create .env file from template:**
```bash
cp .env.example .env
```

**Edit the .env file:**

**macOS/Linux:**
```bash
nano .env
# or use: vim .env, code .env, open -a TextEdit .env
```

**Windows:**
```bash
notepad .env
```

**Add your configuration:**

```bash
# ========================================
# REQUIRED: Groq API Configuration
# ========================================
GROQ_API_KEY=your_actual_groq_api_key_here

# ========================================
# Optional: Model Selection (defaults work well)
# ========================================
GROQ_ASR_MODEL=whisper-large-v3-turbo
GROQ_LLM_MODEL=openai/gpt-oss-20b

# ========================================
# Optional: Kokoro TTS Configuration
# ========================================
KOKORO_VOICE=af_sky
KOKORO_LANG_CODE=a

# ========================================
# Optional: Advanced Settings
# ========================================
ALLOW_FALLBACK_TTS=0
SEED=0
LLM_PRICE_IN_PER_1K=0
LLM_PRICE_OUT_PER_1K=0
```

**⚠️ IMPORTANT:** Replace `your_actual_groq_api_key_here` with your real API key from Step 3!

**Save the file:**
- nano: Press `Ctrl+O`, `Enter`, then `Ctrl+X`
- vim: Press `Esc`, type `:wq`, press `Enter`
- Windows Notepad: File → Save

---

### Step 8: Verify Installation

Run these checks to ensure everything is set up correctly:

**1. Check virtual environment:**
```bash
which python
# macOS/Linux: should show /path/to/Voice_Test_Project/venv/bin/python
# Windows: should show \path\to\Voice_Test_Project\venv\Scripts\python
```

**2. Check Python modules:**
```bash
python -c "import streamlit, groq, kokoro, sounddevice, numpy, torch; print('✅ All modules imported successfully!')"
```

**3. Check espeak-ng:**
```bash
espeak-ng --version
```

**4. Check Groq API key:**
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); key = os.getenv('GROQ_API_KEY'); print('✅ API Key loaded!' if key and key.startswith('gsk_') else '❌ API Key missing or invalid!')"
```

**5. Check project structure:**
```bash
ls -la src/ config/
# Should show: asr_module.py, llm_module.py, tts_module.py, etc.
# Should show: personas/ directory with JSON files
```

**✅ All checks should pass before proceeding to Step 9.**

---

### Step 9: Launch the Application

**Start the Streamlit web application:**
```bash
streamlit run streamlit_app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  For better performance, install the Watchdog module:

  $ xcode-select --install
  $ pip install watchdog
```

**Your browser should automatically open to `http://localhost:8501`**

If it doesn't open automatically:
1. Manually open your web browser
2. Navigate to `http://localhost:8501`

**You should see:** The AI Voice Assistant interface with scenario selection and recording buttons.

---

### Step 10: First Run Test

**Test the complete pipeline:**

1. **Select a scenario** from the dropdown (e.g., "Lost Card")
2. Click **"START RECORDING"** button
3. **Speak clearly**: *"Hi, I lost my credit card yesterday"*
4. Click **"STOP & PROCESS"** button
5. **Wait for processing:**
   - Transcribing... (~1 second)
   - Thinking... (~1-2 seconds)
   - Speaking... (~3-5 seconds)
6. **Listen to the AI's response**

**Expected behavior:**
- You should see your transcribed text in the conversation
- The AI should respond with an empathetic bank agent response
- You should hear the AI speaking through your speakers

**If you hear a proper AI response, congratulations! Setup is complete! 🎉**

---

## Configuration

### Environment Variables Reference

Edit `.env` file to customize:

```bash
# ==================================================
# GROQ API CONFIGURATION
# ==================================================

# Your Groq API key (REQUIRED)
GROQ_API_KEY=gsk_your_key_here

# ASR Model (optional, default: whisper-large-v3-turbo)
# Options: whisper-large-v3-turbo, whisper-large-v3
GROQ_ASR_MODEL=whisper-large-v3-turbo

# LLM Model (optional, default: openai/gpt-oss-20b)
# Options: openai/gpt-oss-20b, llama-3.1-70b-versatile, mixtral-8x7b-32768
GROQ_LLM_MODEL=openai/gpt-oss-20b

# ==================================================
# KOKORO TTS CONFIGURATION
# ==================================================

# Voice selection (default: af_sky)
# Female: af_sky, af_bella, af_heart, af_nicole, af_sarah
# Male: am_adam, am_michael
KOKORO_VOICE=af_sky

# Language code (default: a = American English)
# Options: a (American), b (British)
KOKORO_LANG_CODE=a

# ==================================================
# ADVANCED SETTINGS
# ==================================================

# Allow fallback to macOS 'say' command if Kokoro fails (0 = disabled, 1 = enabled)
ALLOW_FALLBACK_TTS=0

# LLM seed for reproducibility (0 = random, any int = fixed seed)
SEED=0

# Cost tracking (set to actual prices if needed)
LLM_PRICE_IN_PER_1K=0
LLM_PRICE_OUT_PER_1K=0
```

### Available Voices

**Female Voices:**
| Voice | Description | Use Case |
|-------|-------------|----------|
| `af_sky` | Clear, friendly (default) | General purpose, professional |
| `af_bella` | Elegant, sophisticated | Premium services, upscale |
| `af_heart` | Warm, engaging | Empathetic support, care |
| `af_nicole` | Professional, authoritative | Corporate, formal |
| `af_sarah` | Soft, gentle | Calming, reassuring |

**Male Voices:**
| Voice | Description | Use Case |
|-------|-------------|----------|
| `am_adam` | Professional, clear | Business, technical |
| `am_michael` | Deep, authoritative | Leadership, serious topics |

**To change voice:**
1. Edit `.env` file
2. Set `KOKORO_VOICE=af_bella` (or any voice above)
3. Restart the application

---

## Usage

### Streamlit Web UI (Recommended)

**Start the application:**
```bash
streamlit run streamlit_app.py
```

**Using the interface:**

1. **Select Scenario**: Choose from dropdown
   - Lost Card
   - Failed Transfer
   - Locked Account

2. **Start Recording**: Click "START RECORDING" button
   - Status changes to "Recording - Speak now..."
   - Speak your question clearly

3. **Stop & Process**: Click "STOP & PROCESS" button
   - System transcribes your speech (ASR)
   - Generates intelligent response (LLM)
   - Synthesizes voice (TTS)
   - Plays response

4. **Continue Conversation**: Repeat steps 2-3 for follow-up questions

5. **New Conversation**: Click "New Conversation" to reset

**Tips for best results:**
- Speak clearly and at normal pace
- Wait for "Recording" status before speaking
- Minimize background noise
- Use a good microphone (built-in works fine)
- Click STOP immediately after finishing your question

---

### CLI Mode (Advanced)

**For automation and scripting:**

```bash
python main.py --persona card_lost --turns 3
```

**Arguments:**
- `--persona`: Scenario selection
  - `card_lost` - Lost card support
  - `transfer_failed` - Failed transfer support
  - `account_locked` - Locked account support
- `--turns`: Number of conversation turns (default: 3)

**Example:**
```bash
# 5-turn conversation for transfer failure scenario
python main.py --persona transfer_failed --turns 5
```

**CLI features:**
- Automatic voice activity detection (VAD)
- Streaming transcription with partials
- Real-time conversation
- Performance logging to CSV

---

## 📂 Project Structure

```
Voice_Test_Project/
├── src/                          # Core application modules
│   ├── __init__.py              # Package initialization
│   ├── asr_module.py            # Speech recognition (Groq Whisper)
│   ├── llm_module.py            # Language model (Groq LLM)
│   ├── tts_module.py            # Text-to-speech (Kokoro)
│   ├── simple_voice_handler.py  # Manual recording pipeline
│   ├── voice_client.py          # Auto VAD pipeline (CLI mode)
│   ├── state_manager.py         # Conversation state tracking
│   ├── logger.py                # Logging utilities
│   └── feedback.py              # Post-conversation evaluation
│
├── config/                       # Configuration files
│   ├── __init__.py
│   └── personas/                # AI behavior definitions
│       ├── card_lost.json       # Lost card scenario
│       ├── transfer_failed.json # Failed transfer scenario
│       └── account_locked.json  # Locked account scenario
│
├── logs/                         # Performance logs
│   ├── .gitkeep
│   └── latency_log.csv         # Auto-generated metrics
│
├── streamlit_app.py             # Web UI application (main entry)
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── .env                         # Your configuration (create this)
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
└── venv/                        # Virtual environment (created by you)
```

**Key Files:**
- **`streamlit_app.py`**: Main web interface
- **`main.py`**: CLI interface for automation
- **`src/simple_voice_handler.py`**: Core pipeline logic
- **`config/personas/*.json`**: AI behavior definitions
- **`.env`**: Your API keys and configuration

---

## Scenarios

The system includes three pre-configured bank support scenarios:

### Lost Card Support

**Persona File:** `config/personas/card_lost.json`

**AI Behavior:**
- **Tone**: Empathetic and reassuring
- **Priority**: Security and quick action
- **Workflow**:
  1. Acknowledge customer concern warmly
  2. Ask for security verification (last 4 digits)
  3. Confirm immediate card blocking
  4. Explain replacement timeline (5-7 business days)
  5. Offer digital card alternatives
  6. Provide fraud monitoring information

**Example interaction:**
- **User**: "I lost my credit card yesterday"
- **AI**: "I'm really sorry to hear that. Let me help you secure your account right away. Can you confirm the last 4 digits of your card for security?"

---

### Failed Transfer Support

**Persona File:** `config/personas/transfer_failed.json`

**AI Behavior:**
- **Tone**: Solution-focused and efficient
- **Priority**: Fast resolution
- **Workflow**:
  1. Acknowledge frustration quickly
  2. Request transfer details (amount, recipient, time)
  3. Identify issue (balance, limits, recipient problems)
  4. Provide immediate solution
  5. Suggest alternatives if needed

**Example interaction:**
- **User**: "My transfer to Sarah didn't go through"
- **AI**: "I'm sorry you're having trouble—let's get this sorted quickly. Can you tell me the amount you tried to send and when you attempted the transfer?"

---

### Locked Account Support

**Persona File:** `config/personas/account_locked.json`

**AI Behavior:**
- **Tone**: Reassuring and educational
- **Priority**: Security explanation and unlock
- **Workflow**:
  1. Reassure it's a security measure
  2. Explain trigger (travel, unusual activity)
  3. Verify customer identity
  4. Unlock account
  5. Educate on prevention

**Example interaction:**
- **User**: "I can't log into my account anymore"
- **AI**: "Don't worry, this is a security measure to protect your account. I can help unlock it. Have you traveled recently or made any unusual transactions?"

---

### Customizing Personas

Edit JSON files in `config/personas/` to customize AI behavior:

```json
{
  "name": "Lost Card Support",
  "scenario": "Card Lost",
  "system_prompt": "You are an empathetic bank support agent helping a customer who lost their card. Be warm, security-focused, and provide clear next steps..."
}
```

**Fields:**
- `name`: Display name for the persona
- `scenario`: Short scenario description
- `system_prompt`: Detailed instructions for the LLM

---

## Performance Metrics

### Typical Latency (per turn)

| Component | Time Range | Average | Notes |
|-----------|------------|---------|-------|
| **Recording** | User-controlled | Variable | Until user clicks STOP |
| **ASR** | 500-1500ms | ~800ms | Depends on audio length |
| **LLM** | 400-2000ms | ~500ms | Depends on response length |
| **TTS** | 2000-10000ms | ~5000ms | Depends on response length |
| **Total** | 3-13 seconds | ~6 seconds | Excluding recording time |

### Resource Usage

| Resource | Usage | Notes |
|----------|-------|-------|
| **Memory** | 300-500MB | Includes loaded models |
| **CPU** | Moderate | Peaks during TTS synthesis |
| **Network** | ~50-200KB/turn | Only for ASR and LLM API calls |
| **Storage** | ~500MB | Models and dependencies |

### Performance Tips

1. **Faster responses**: Use shorter questions
2. **Better accuracy**: Speak clearly with minimal background noise
3. **Reduce latency**: Use a faster internet connection
4. **Lower memory**: Close other applications

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "No module named 'src'"

**Cause:** Running from wrong directory

**Solution:**
```bash
# Make sure you're in the project root
cd /path/to/Voice_Test_Project

# Then run
streamlit run streamlit_app.py
```

---

#### Issue 2: "No audio recorded"

**Causes:**
- Microphone not working
- Wrong microphone selected
- Didn't click STOP button
- No permission for microphone

**Solutions:**
- **Check microphone**: Test with system recorder
- **Check permissions**: Allow microphone access in System Preferences/Settings
- **Click STOP**: Must click STOP & PROCESS after speaking
- **Try different mic**: Select different device in system settings

---

#### Issue 3: "API key error" or "401 Unauthorized"

**Cause:** Invalid or missing Groq API key

**Solution:**
```bash
# Check if .env file exists
ls -la .env

# Check if API key is set
cat .env | grep GROQ_API_KEY

# If empty or wrong, edit .env
nano .env
# Add: GROQ_API_KEY=your_actual_key_here
```

---

#### Issue 4: "espeak-ng not found"

**Cause:** espeak-ng not installed or not in PATH

**Solution:**
```bash
# macOS
brew install espeak-ng

# Ubuntu/Debian
sudo apt-get install espeak-ng

# Verify
espeak-ng --version
```

---

#### Issue 5: "'ASRClient' object has no attribute 'transcribe'"

**Cause:** Code mismatch or outdated files

**Solution:**
```bash
# Make sure all files are up to date
# Check that src/asr_module.py has transcribe_wav_bytes method
# Restart the application
```

---

#### Issue 6: "Port 8501 is already in use"

**Cause:** Another Streamlit app is running

**Solution:**
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use different port
streamlit run streamlit_app.py --server.port 8502
```

---

#### Issue 7: Slow or no response from AI

**Causes:**
- Poor internet connection
- Groq API issues
- Rate limiting

**Solutions:**
- Check internet connection
- Wait a moment and try again
- Check [Groq Status](https://status.groq.com/)
- Verify API key quota

---



