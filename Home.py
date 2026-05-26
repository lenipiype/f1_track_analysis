import streamlit as st

st.set_page_config(
    page_title="DSP Exercises",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏎️ Digital Signal Processing – Exercise Collection")
st.markdown(
    """
    Welcome to the DSP exercise collection. Use the sidebar to navigate between exercises.

    | Exercise | Topic | Status |
    |---|---|---|
    | Exercise 1 | Time Domain Analysis | ✅ Complete |
    | Exercise 2 | Frequency Domain Analysis| ✅ Complete |
    | Exercise 3 | Time-Frequency Analysis| ✅ Complete |

    ---
    *Digital Signal Processing, 2026*
    """
)
