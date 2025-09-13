import streamlit as st


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception as e:
            return f"Could not load {path}: {e}"


def render_about_panel():
    with st.expander("About", expanded=False):
        st.markdown(
            """
            **EVM Assistant**

            - Purpose: Compute Earned Value Management (EVM) metrics from a portfolio CSV, assess risks, and provide summaries with integrated Q&A.
            - Modes: Fast (local compute + 1 summary) or Agentic (multi‑agent orchestration).
            - Privacy: No persistence by default; calculations are local, and Q&A is restricted to your uploaded data.
            - Models: Configurable default and summary models; provider auto‑detected.

            View the project docs below:
            """
        )

        # Quick links to GitHub repo and docs
        repo_url = "https://github.com/bashiraziz/EVM-ASSISTANT"
        readme_url = repo_url + "/blob/main/README.md"
        deploy_url = repo_url + "/blob/main/DEPLOY.md"
        st.markdown(
            f"- Repo: [{repo_url}]({repo_url})\n"
            f"- README: [{readme_url}]({readme_url})\n"
            f"- DEPLOY: [{deploy_url}]({deploy_url})"
        )

        with st.expander("View README.md", expanded=False):
            st.markdown(_read_text("README.md"))

        with st.expander("View DEPLOY.md", expanded=False):
            st.markdown(_read_text("DEPLOY.md"))


__all__ = ["render_about_panel"]
