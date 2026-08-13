import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


BACKEND_URL = "http://localhost:8000/chat"


def post_chat(backend_url, prompt, keep_ratio):
    payload = {
        "message": prompt,
        "keep_ratio": keep_ratio,
    }
    request = Request(
        backend_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


st.set_page_config(page_title="Router Chat", layout="centered")
st.title("Router Chat")

with st.sidebar:
    st.subheader("Settings")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)
    keep_ratio = st.slider("Compression keep ratio", 0.1, 1.0, 0.6, 0.05)
    show_debug = st.toggle("Show routing details", value=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("debug") and show_debug:
            with st.expander("Routing details"):
                st.json(message["debug"])

prompt = st.chat_input("Send a prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Compressing, routing, and calling the selected model..."):
            try:
                data = post_chat(backend_url, prompt, keep_ratio)
                reply = data["reply"]
                debug = {
                    "selected_router_model": data["selected_router_model"],
                    "selected_openrouter_model": data["selected_openrouter_model"],
                    "features": data["features"],
                    "compressed_prompt": data["compressed_prompt"],
                    "reduction": data["reduction"],
                    "dropped_tokens": data["dropped_tokens"],
                }
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                reply = f"Backend request failed with {exc.code}: {body}"
                debug = None
            except (URLError, TimeoutError) as exc:
                reply = f"Backend request failed: {exc}"
                debug = None

        st.markdown(reply)
        if debug and show_debug:
            with st.expander("Routing details"):
                st.json(debug)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "debug": debug,
        }
    )
