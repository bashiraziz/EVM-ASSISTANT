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
        </style>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["inject_theme"]

