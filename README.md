# AI Video Factory 🤖🎥

A full-stack, AI-powered video automation platform that transforms topics and scripts into viral-ready videos using OpenAI, Gemini 1.5 Pro, and Groq.

## 🚀 Features

- **Multi-Engine Scripting**: Uses Gemini 1.5 Pro for creative storytelling and Groq (Llama 3) as a high-speed fallback.
- **AI Voiceovers**: Integrated with OpenAI TTS for realistic narration.
- **Visual Generation**: DALL-E 3 generated cinematic visuals for every scene.
- **Automated Video Editing**: FFmpeg-powered pipeline to stitch images, audio, and transitions.
- **Smart Dashboard**: AI Orchestrator to handle system commands and dark/light mode.
- **Cloud Sync**: Direct upload to Google Drive.
- **Admin Controls**: User management and access request system.

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Tailwind CSS, Jinja2 Templates
- **AI Models**: 
  - GPT-4o (Orchestration)
  - Gemini 1.5 Pro (Creative Scripting)
  - Llama 3 via Groq (Fast Fallback)
  - DALL-E 3 (Image Generation)
  - OpenAI TTS (Voice)
- **Processing**: FFmpeg
- **Storage**: Local + Google Drive API

## 📦 Installation

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`:
   ```env
   SECRET_KEY=your_secret_key
   GEMINI_API_KEY=your_gemini_key
   OPENAI_API_KEY=your_openai_key
   GROQ_API_KEY=your_groq_key
   OWNER_EMAILS=admin@example.com
   OWNER_MOBILES=+91...
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## 🛡️ Security

- JWT-based authentication with 24h expiration.
- Simulated OTP verification for authorized owners.
- Strict asset purging (Storage & DB cleanup before every new generation).
- Fail-safe PDF generation if video rendering fails.

## 📄 License

MIT License
