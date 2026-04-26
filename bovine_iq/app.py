import streamlit as st
import pandas as pd
import os
import random
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables (like GEMINI_API_KEY) from .env
load_dotenv()

from data.ingestion import DataIngestion
from agent.core import BovineIQAgent
from monitoring.alerts import run_herd_monitoring

# Initialize Singletons
@st.cache_resource
def get_data_engine():
    return DataIngestion()
    
def render_message_with_image(content: str):
    import re
    match = re.search(r'\[SHOW_IMAGE:\s*(.*?)\]', content)
    if match:
        img_path = match.group(1).strip()
        clean_text = content.replace(match.group(0), "").strip()
        if clean_text:
            st.markdown(clean_text)
        try:
            st.image(img_path, caption="Live Feed", use_container_width=True)
        except Exception:
            st.error(f"Failed to load image from path: {img_path}")
    else:
        st.markdown(content)
    
@st.cache_resource
def get_agent():
    return BovineIQAgent()

ingestion = get_data_engine()
agent = get_agent()

st.set_page_config(
    page_title="BovineIQ - Veterinary Assistant",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern premium farm/vet dark aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Main Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    h1 { font-size: 2.8rem; background: -webkit-linear-gradient(45deg, #34d399, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    /* Metrics box */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        color: #34d399; /* emerald-400 */
        text-shadow: 0 0 10px rgba(52, 211, 153, 0.2);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Cards and Containers (Glassmorphism) */
    .css-1r6slb0, .css-1n76uvr, div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(52, 211, 153, 0.3);
    }
    
    /* Warning/Alert banners */
    .stAlert {
        border-radius: 12px;
        border: none;
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(8px);
    }
    
    /* Custom divider */
    hr {
        border-color: rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

def render_dashboard():
    st.title("🐄 BovineIQ Farm Dashboard")
    st.markdown("Real-time overview of herd health and production metrics.")
    
    summary = ingestion.get_herd_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Herd", f"{summary.get('total_cows', 16)} Cows")
    with col2:
        st.metric("Health Events Logged", str(summary.get('total_health_events', 0)))
    with col3:
        st.metric("Total Milk Yield Today", f"{summary.get('total_milk_today', 0)} kg")
    with col4:
        st.metric("Avg THI", "72 (Mild Stress)", "+1.5")
        
    st.markdown("---")
    st.subheader("⚠️ Critical Alerts & Anomaly Detection")
    
    live_alerts = run_herd_monitoring(ingestion)
    if not live_alerts:
        st.success("All Cows are operating within normal parameters. No anomalies detected.")
    else:
        for alert in live_alerts:
            # Render visually based on severity
            msg = f"**{alert['cow_id']}** — {alert['type']}: {alert['msg']}"
            if alert['level'] == "CRITICAL":
                st.error(msg)
            elif alert['level'] == "HIGH":
                st.warning(msg)
            else:
                st.info(msg)

    st.subheader("Herd Health Score Distribution")
    # Placeholder chart
    chart_data = pd.DataFrame(
        {"Bands": ["Healthy (0-25)", "Watch (25-50)", "At-Risk (50-75)", "Critical (>75)"],
         "Count": [10, 4, 1, 1]}
    )
    st.bar_chart(chart_data.set_index("Bands"))

def process_agent_response(response):
    """Processes the dictionary or string response from the agent and updates session state."""
    if type(response) is dict:
        if response.get("status") == "pending":
            st.session_state.pending_agent_state = response
            code_str = response.get("tool_call", {}).get("args", {}).get("code", "")
            msg = f"""**⚠️ Code Execution Requested:**

```python
{code_str}
```

_Please approve or reject this execution below._"""
            st.session_state.messages.append({"role": "assistant", "content": msg})
            return msg
        elif response.get("status") == "done":
            msg = response.get("content", "")
            st.session_state.messages.append({"role": "assistant", "content": msg})
            return msg
    else:
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
        return str(response)

def render_vet_assistant():
    st.title("🩺 BovineIQ Vet Assistant")
    st.markdown("Chat with the AI veterinary agent powered by the MMCOWS dataset and comprehensive vet literature.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to BovineIQ. How can I assist you with herd health today?"}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_message_with_image(message["content"])

    # React to user input
    
    # If we are waiting for code approval, show buttons and stop
    if st.session_state.get("pending_agent_state") is not None:
        with st.chat_message("assistant"):
            st.markdown("### ⚠️ Authorization Required")
            st.markdown("The BovineIQ agent intends to execute the Python script shown above. Do you authorize this action?")
            col1, col2 = st.columns(2)
            if col1.button("✅ Approve & Execute"):
                state = st.session_state.pending_agent_state
                st.session_state.pending_agent_state = None
                with st.spinner("Executing and analyzing results..."):
                    response = agent.resume(state, approved=True)
                    process_agent_response(response)
                st.rerun()
            if col2.button("❌ Reject Workflow"):
                state = st.session_state.pending_agent_state
                st.session_state.pending_agent_state = None
                with st.spinner("Informing agent of cancellation..."):
                    response = agent.resume(state, approved=False)
                    process_agent_response(response)
                st.rerun()
        return  # Prevent chat input from rendering

    if prompt := st.chat_input("Ask about a cow's data or a veterinary protocol..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("BovineIQ is analyzing..."):
            # Get response from BovineIQ AI Agent
            try:
                formatted_history = []
                for msg in st.session_state.messages[:-1][-5:]:
                    role = "human" if msg["role"] == "user" else "ai"
                    formatted_history.append((role, msg["content"]))
                    
                response = agent.query(user_input=prompt, history=formatted_history)
            except Exception as e:
                response = f"Agent Error (check API key): {str(e)}"
                
        # Process and save
        process_agent_response(response)
        st.rerun()

def render_cow_profiles():
    st.title("📋 Individual Cow Profiles")
    selected_cow = st.selectbox("Select Cow ID", [f"C{i:02d}" for i in range(1, 17)])
    
    # Fetch real stats
    stats = ingestion.get_latest_stats(selected_cow)
    
    st.subheader(f"Data for {selected_cow}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sensor Status:** 🟢 Active")
        st.markdown("**Health Events Logged:** " + str(stats.get("health_events", 0)))
        sensors = stats.get("other_sensors", {})
        st.markdown(f"**Ankle Sensor:** {sensors.get('ankle', 'Not Found')}")
        st.markdown(f"**IMMU Sensor:** {sensors.get('immu', 'Not Found')}")
    with col2:
        st.markdown(f"**7-Day Avg Daily Yield:** {stats.get('avg_milk_7d', 'N/A')} kg")
        st.markdown(f"**Latest CBT:** {stats.get('latest_cbt', 'N/A')} °C")
        st.markdown(f"**UWB Location:** {sensors.get('uwb', 'Not Found')}")
        st.markdown(f"**THI Environment:** {sensors.get('thi', 'Not Found')}")
    
    st.markdown("### 📈 Sensor Analytics")
    tab1, tab2 = st.tabs(["Core Body Temperature (CBT)", "Milk Yield (7-Day)"])
    
    with tab1:
        cbt_df = ingestion.load_cbt(selected_cow)
        if cbt_df is not None and not cbt_df.empty and 'temperature_C' in cbt_df.columns:
            # Plot the last 300 data points for performance
            chart_data = cbt_df['temperature_C'].tail(300).reset_index(drop=True)
            st.line_chart(chart_data, height=300)
            
            latest_temp = cbt_df['temperature_C'].iloc[-1]
            if latest_temp > 39.5:
                st.error("⚠️ Anomaly Detected: Elevated Temperature (Fever Risk)")
        else:
            st.warning("No CBT data available for this cow.")
            
    with tab2:
        milk_df = ingestion.load_milk(selected_cow)
        if milk_df is not None and not milk_df.empty and 'milk_kg' in milk_df.columns:
            st.bar_chart(milk_df.set_index('timestamp')['milk_kg'].tail(7) if 'timestamp' in milk_df.columns else milk_df['milk_kg'].tail(7), height=300)
        else:
            st.warning("No Milk Yield data available.")



def main():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/10708/10708085.png", width=100) # Placeholder logo
        st.title("BovineIQ")
        st.markdown("Navigation")
        page = st.radio("Go to", ["Dashboard", "Vet Assistant", "Cow Profiles", "Data Settings"])
        
    if page == "Dashboard":
        render_dashboard()
    elif page == "Vet Assistant":
        render_vet_assistant()
    elif page == "Cow Profiles":
        render_cow_profiles()
    elif page == "Data Settings":
        st.title("⚙️ Data Settings")
        st.write("Configure MMCOWS dataset directories, vector store connections, and LLM preferences here.")

if __name__ == "__main__":
    main()
