from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os


#image to text part
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("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")


def image2text(url):

    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return(text)


#llm part

def generate_story(scenario):
    template = """
    You are a story teller
    You can generate a short story based on a simple narrative, the story should be no more than 20 words;
    
    CONTEXT: {scenario}
    STORY: 
    """

    prompt = PromptTemplate(template = template, input_variables = ["scenario"])

    story_llm = LLMChain(llm=OpenAI(
        model_name = "gpt-3.5-turbo", tempereture = 1), prompt=prompt, verbose = True
    )

    story = story_llm.predict(scenario=scenario)

    print(story)
    return story

scenario = image2text("image1.jpg")
story = generate_story(scenario)

#test to speech

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Autherization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {    
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)
