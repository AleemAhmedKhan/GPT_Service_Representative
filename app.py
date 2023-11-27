from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os

load_dotenv(find_dotenv())

ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

def get_response_from_csr(human_input):
    template ="""
    you are as a customer service representative, now let's handle the customer's inquiry:
    1/ Your name is Alex, a customer service representative at XYZ Company.
    2/ You aim to provide helpful and accurate information to customers.
    3/ Be professional, polite, and address the customer's needs effectively.

    {history}
    Customer: {human_input}
    Alex:    
    """
    
    prompt = PromptTemplate(
        input_variables=("history", "customer_inputs"),
        template=template 
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output 

def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_nonolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post('http://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDqBikWAm?optimize_streaming_latency=0', json=payload, headers=headers)

    if response.status_code == 200 and response.content:
        with open('audio.mp3', 'wb') as f:
            f.write(response.content)
        playsound('audio.mp3')

# Building the UI with Flask
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    customer_input = request.form['customer_input']
    message = get_response_from_csr(customer_input)
    get_voice_message(message)
    return message

if __name__ == "__main__":
    app.run(debug=True)
