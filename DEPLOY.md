Side‑by‑Side Hosting Guide (Streamlit + Gradio)

Overview
This project can expose two UIs from the same codebase:
1) Streamlit: the full-featured EVM Assistant you’ve been using
2) Gradio (optional): a parallel “Blocks” demo that calls the same compute/agent code

Both can be hosted from this repository on common platforms. Below are the recommended options and exact steps.

Repository Prerequisites
- Entry points:
  - Streamlit: evms-agents-app.py
  - Gradio: app_gradio.py (add this file if not present; see scaffold in notes below)
- Procfile: provided (Streamlit)
- Dockerfile: provided (defaults to Streamlit)
- requirements.txt: include streamlit, gradio, agents, openai, litellm, python-dotenv, pandas, numpy
- Environment variables (set per service):
  - OPENAI_API_KEY or GEMINI_API_KEY (depending on your model provider)
  - Optional: AGENTS_DEFAULT_MODEL, AGENTS_SUMMARY_MODEL, LITELLM_LOG, etc.

Option A — Streamlit Community Cloud + Hugging Face Spaces
Streamlit (Streamlit Cloud)
1. Push repo to GitHub
2. Go to https://share.streamlit.io, connect GitHub
3. Select this repo; entry file: evms-agents-app.py
4. In App settings → Secrets, add API keys (OPENAI_API_KEY / GEMINI_API_KEY)
5. Deploy

Gradio (Hugging Face Spaces)
1. Create a Space → SDK: “Gradio”, set app_file: app_gradio.py
2. Add repository secrets for API keys
3. Include requirements.txt (gradio≥4.0 etc.)
4. Deploy

Option B — Render (two Web Services)
Service 1 (Streamlit)
1. New Web Service → from repo
2. Environment: Python
3. Start Command: streamlit run evms-agents-app.py --server.port $PORT --server.address 0.0.0.0
4. Set environment variables (API keys)
5. Deploy

Service 2 (Gradio)
1. New Web Service → from same repo
2. Environment: Python
3. Start Command: python app_gradio.py
4. Set environment variables (API keys)
5. Deploy

Option C — Google Cloud Run (two services, one image)
Build image (once)
  gcloud builds submit --tag gcr.io/PROJECT/evm-assistant

Deploy Streamlit (default CMD from Dockerfile)
  gcloud run deploy evm-streamlit \
    --image gcr.io/PROJECT/evm-assistant \
    --platform managed --allow-unauthenticated --region REGION

Deploy Gradio (override command)
  gcloud run deploy evm-gradio \
    --image gcr.io/PROJECT/evm-assistant \
    --command python --args app_gradio.py \
    --platform managed --allow-unauthenticated --region REGION

Set env vars (keys) on both services in Cloud Run console.

Optional: Docker RUN_TARGET switch
You can drive both UIs from the same image by setting an env switch.

  # in Dockerfile (concept)
  ENV RUN_TARGET=streamlit
  CMD bash -lc 'if [ "$RUN_TARGET" = "gradio" ]; then python app_gradio.py; \
    else streamlit run evms-agents-app.py --server.port ${PORT:-8080} --server.address 0.0.0.0; fi'

Then deploy two services with different RUN_TARGET values.

Option D — Heroku (two apps from one repo)
App 1 (Streamlit)
1. Use existing Procfile: web: streamlit run evms-agents-app.py --server.port $PORT --server.address 0.0.0.0
2. Set config vars (API keys)
3. Deploy from GitHub or via git push heroku main

App 2 (Gradio)
Option 1: Add a Procfile.gradio in a separate branch; or
Option 2: In the app’s settings, create a Procfile with: web: python app_gradio.py
Set the same API keys in Config Vars, then deploy.

Environment Checklist (all platforms)
- Port binding:
  - Streamlit/Gradio must bind to 0.0.0.0:$PORT (already in Procfile/Dockerfile)
- Secrets/keys (set per app/service):
  - OPENAI_API_KEY and/or GEMINI_API_KEY
  - Optional model config: AGENTS_DEFAULT_MODEL, AGENTS_SUMMARY_MODEL
- Tracing/logging: tracing_disabled=True is already used for stability
- Memory/time: choose instance sizes that suit the CSV size and model latency

Gradio Scaffold (app_gradio.py, minimal)
If you don’t have it yet, create app_gradio.py with a Blocks app that calls your compute and agents. The assistant provided a ready-to-use scaffold in prior conversation; it:
- Reuses compute_portfolio_for_ui and your agents via Runner.run
- Shows a summary, metrics table, totals JSON, suggestions, and “Ask Rowshni” Q&A
- Uses gr.State for per-user session and gr.Progress for status

Troubleshooting
- If one app starts but the other returns 502:
  - Verify Start Command / CMD is correct for each service
  - Confirm $PORT is injected (Render/Heroku/Cloud Run do this automatically)
  - Check requirements.txt includes gradio and streamlit
- If model calls fail:
  - Verify API keys are present in that service’s env
  - Confirm model identifiers (LiteLLM or OpenAI) are valid for the provider

