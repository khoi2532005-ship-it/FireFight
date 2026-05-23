import streamlit as st

def show_header():
    st.markdown(
        """
        <div style='text-align: center; padding: 0.5rem 0 1rem 0;'>
            <span style='font-size: 2rem;'>🔥</span>
            <span style='font-size: 1.5rem; font-weight: 600; margin-left: 8px;'>Fire Fight</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")