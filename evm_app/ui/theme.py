import streamlit as st


def inject_theme():
    st.markdown(
        """
        <style>
        :root{
          --acc1:#0b6cf0;
          --acc2:#18c29c;
          --panel:#0f1b2e;
        }
        .trace-card{background:linear-gradient(135deg,rgba(11,108,240,.18),rgba(24,194,156,.18));border:1px solid rgba(255,255,255,.08);} 
        .table-head th{background:linear-gradient(135deg,rgba(11,108,240,.35),rgba(24,194,156,.35));color:#fff;border-bottom:1px solid rgba(255,255,255,.08)!important}
        .totals-banner{background:linear-gradient(135deg,var(--acc1),var(--acc2));color:#fff}
        /* Markdown typography tweaks for readability */
        h1, .stMarkdown h1{font-size:1.6rem;}
        h2, .stMarkdown h2{font-size:1.3rem;}
        h3, .stMarkdown h3{font-size:1.1rem;}
        /* Table polish inside markdown blocks */
        .stMarkdown table{border-collapse:collapse;border:1px solid rgba(255,255,255,.12)}
        .stMarkdown th, .stMarkdown td{border:1px solid rgba(255,255,255,.12);padding:6px 10px}
        .stMarkdown th{background:rgba(255,255,255,.06)}
        /* Card wrapper for narrative report */
        .report-card{background:radial-gradient(1200px 400px at 0% 0%, rgba(11,108,240,.12), transparent),
                     radial-gradient(1200px 400px at 100% 0%, rgba(24,194,156,.10), transparent);
                     border:1px solid rgba(255,255,255,.10); border-radius:12px; padding:12px 14px;}
        /* Primary CTA wrapper to target the Run button only */
        .primary-cta .stButton>button{
            font-weight:700; border-radius:999px; padding:10px 18px;
            background:linear-gradient(90deg,var(--acc1),var(--acc2));
            color:#fff; border:0; box-shadow:0 2px 12px rgba(11,108,240,.25);
        }
        .primary-cta .stButton>button:hover{filter:brightness(1.02)}
        .primary-cta .stButton>button:focus{outline:2px solid rgba(24,194,156,.6)}

        /* Highlight the expander summary for "View latest answer" like a pill */
        .hl-exp [data-testid="stExpander"] > details > summary{
            background:linear-gradient(90deg,var(--acc1),var(--acc2));
            color:#fff; border:0; border-radius:999px; padding:8px 14px; font-weight:700;
            box-shadow:0 2px 12px rgba(11,108,240,.18);
        }
        .hl-exp [data-testid="stExpander"] > details > summary:hover{filter:brightness(1.02)}
        .hl-exp [data-testid="stExpander"] > details > summary p{color:#fff; margin:0}
        </style>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["inject_theme"]
