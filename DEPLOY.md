Deployment Guide (Streamlit only)

Overview
This project uses a single Streamlit UI. All Gradio files and references have been removed.

Entry point
- Streamlit app file: evms-agents-app.py

Requirements
- requirements.txt includes Streamlit and supporting libs (no Gradio).
- Environment variables:
  - OPENAI_API_KEY or GEMINI_API_KEY (depending on provider)
  - Optional: AGENTS_DEFAULT_MODEL, AGENTS_SUMMARY_MODEL, LITELLM_LOG

Option A — Streamlit Community Cloud
1. Push repo to GitHub
2. Go to https://share.streamlit.io and connect GitHub
3. Select this repo; entry file: evms-agents-app.py
4. Add Secrets (API keys)
5. Deploy

Option B — Render
1. New Web Service ? from repo
2. Start Command: streamlit run evms-agents-app.py --server.port $PORT --server.address 0.0.0.0
3. Set environment variables (API keys)
4. Deploy

Option C — Google Cloud Run
1. Build an image that runs the Streamlit command by default, or use a Procfile-based buildpack.
2. Ensure the process binds to 0.0.0.0:$PORT.

Option D — Heroku
1. Procfile:
   web: streamlit run evms-agents-app.py --server.port $PORT --server.address 0.0.0.0
2. Set config vars (API keys) and deploy.

Environment Checklist
- Port binding: 0.0.0.0:$PORT on platforms that inject PORT
- Secrets: OPENAI_API_KEY and/or GEMINI_API_KEY
- Optional model config: AGENTS_DEFAULT_MODEL, AGENTS_SUMMARY_MODEL
- Keep tracing disabled in production for stability

Local Run
  streamlit run evms-agents-app.py
