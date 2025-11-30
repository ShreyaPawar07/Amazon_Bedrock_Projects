import streamlit as st #all streamlit commands will be available through the "st" alias
import json
import boto3

session = boto3.Session()

bedrock = session.client(service_name='bedrock-runtime') #creates a Bedrock client

bedrock_model_id = "us.amazon.nova-lite-v1:0" #set the foundation model

# prompt = "What is the largest city in New Hampshire?" #the prompt to send to the model

def summarize_text(prompt,max_tokens,temperature):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": f"You are a text summarizer please summarize given text {prompt}"}
            ]
        }
    ]

    body = json.dumps({
        "schemaVersion": "messages-v1",
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "topP": 0.5,
            "topK": 20,
            "temperature": temperature
        }
    }) #build the request payload

    response = bedrock.invoke_model(body=body, modelId=bedrock_model_id, accept='application/json', contentType='application/json') #send the payload to Amazon Bedrock

    response_body = json.loads(response.get('body').read()) # read the response

    response_text = response_body["output"]["message"]["content"][0]["text"] #extract the text from the JSON response

    return response_text



st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="centered",
)


st.markdown(
    """
    <style>
        .main {
            background: #fffbf5; /* light peach body background */
            color: #1f2933;
        }
        section[data-testid="stSidebar"] {
            background-color: #fed7aa; /* light orange background for sidebar */
            border-right: 1px solid #f97316;
        }
        /* Slider track & thumb styling (tuned for light orange sidebar) */
        .stSlider [data-baseweb="slider"] > div {
            background-color: #fffbeb; /* light track background */
        }
        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background-color: #22c55e; /* bright thumb color */
            box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.4);
        }
        .stSlider [data-baseweb="slider"] > div > div {
            background: linear-gradient(90deg, #22c55e, #4ade80);
        }
        .summary-box {
            background-color: #020617;
            border-radius: 0.75rem;
            padding: 1.25rem 1.5rem;
            border: 1px solid #1f2937;
        }
        .stTextArea textarea {
            background-color: #f3f4f6 !important; /* light gray background */
            color: #111827 !important;
            border-radius: 0.75rem !important;
            border: 2px solid #ec4899 !important; /* pink border */
        }
        /* Glowy primary button */
        .stButton button {
            background: linear-gradient(135deg, #22c55e, #4ade80);
            color: #0f172a;
            border-radius: 999px;
            border: none;
            box-shadow: 0 0 14px rgba(34, 197, 94, 0.7);
            font-weight: 600;
            transition: box-shadow 0.2s ease, transform 0.2s ease, filter 0.2s ease;
        }
        .stButton button:hover {
            box-shadow: 0 0 22px rgba(34, 197, 94, 0.95);
            transform: translateY(-1px);
            filter: brightness(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center; color: #be185d;'>✨ Text Summarizer ✨</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #9ca3af;'>Paste your content, tune the settings, and get a clean summary in seconds.</p>",
    unsafe_allow_html=True,
)

# Sidebar controls for model parameters
with st.sidebar:
    st.header("⚙️ Generation Settings")

    # Slider for max tokens
    max_tokens = st.slider(
        "Max Tokens",
        min_value=16,
        max_value=4096,
        value=1024,
        step=16,
        help="Controls the maximum length of the generated summary."
    )
    # `max_tokens` is the variable that holds the user's selected max token value.

    # Slider for temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="Controls randomness: lower is more focused, higher is more creative."
    )
    # `temperature` is the variable that holds the user's selected temperature value.

st.markdown(
    "<h3 style='color: #38bdf8;'>🧾 Input Text</h3>",
    unsafe_allow_html=True,
)

# Text input from the user
text_to_summarize = st.text_area(
    "Enter text to summarize",
    height=200,
    placeholder="Paste or type your text here...",
)

if st.button("Summarize"):
    
    summary = summarize_text(
        text_to_summarize,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # st.markdown(
    #     "<h4 style='color: #4ade80;'>📌 Summary</h4>",
    #     unsafe_allow_html=True,
    # )
    with st.container():
        st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
        st.write(summary)
        st.markdown("</div>", unsafe_allow_html=True)

