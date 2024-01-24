from typing import Dict, List
from langchain.llms import Replicate
from langchain.memory import ChatMessageHistory
from langchain.schema.messages import get_buffer_string
import os

# Get a free API key from https://replicate.com/account/api-tokens
os.environ["REPLICATE_API_TOKEN"] = "USE YOUR OWN GENERATED KEY"

LLAMA2_70B_CHAT = "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48"
LLAMA2_13B_CHAT = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"

# We'll default to the smaller 13B model for speed; change to LLAMA2_70B_CHAT for more advanced (but slower) generations
DEFAULT_MODEL = LLAMA2_13B_CHAT
def completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    llm = Replicate(
        model=model,
        model_kwargs={"temperature": temperature,"top_p": top_p, "max_new_tokens": 1000}
    )
    return llm(prompt)

def chat_completion(
    messages: List[Dict],
    model = DEFAULT_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        elif message["role"] == "assistant":
            history.add_ai_message(message["content"])
        else:
            raise Exception("Unknown role")
    return completion(
        get_buffer_string(
            history.messages,
            human_prefix="USER",
            ai_prefix="ASSISTANT",
        ),
        model,
        temperature,
        top_p,
    )

def assistant(content: str):
    return { "role": "assistant", "content": content }

def user(content: str):
    return { "role": "user", "content": content }

def complete_and_print(prompt: str, model: str = DEFAULT_MODEL):
    print(f'==============\n{prompt}\n==============')
    response = completion(prompt, model)
    print(response, end='\n\n')
    
#complete_and_print("The typical color of the sky is: ")
#complete_and_print("which model version are you?")
response = chat_completion(messages=[
    user("My favorite color is blue."),
    assistant("That's great to hear!"),
    user("What is my favorite color?"),
])
print(response)

'''def print_tuned_completion(temperature: float, top_p: float):
    response = completion("Write a haiku about llamas", temperature=temperature, top_p=top_p)
    print(f'[temperature: {temperature} | top_p: {top_p}]\n{response.strip()}\n')

print_tuned_completion(0.01, 0.01)
print_tuned_completion(0.01, 0.01)
# These two generations are highly likely to be the same

print_tuned_completion(1.0, 1.0)
print_tuned_completion(1.0, 1.0)
# These two generations are highly likely to be different
'''
#Detailed, explicit instructions produce better results than open-ended prompts:

#complete_and_print(prompt="Describe quantum physics in one short sentence of no more than 12 words")
# Returns a succinct explanation of quantum physics that mentions particles and states existing simultaneously.

#complete_and_print(prompt="Describe quantum physics in one short sentence of no more than 12 words")
# Returns a succinct explanation of quantum physics that mentions particles and states existing simultaneously.

#complete_and_print("Explain the latest advances in large language models to me.")
# More likely to cite sources from 2017

#complete_and_print("Explain the latest advances in large language models to me. Always cite your sources. Never cite sources older than 2020.")
# Gives more specific advances and only cites sources from 2020

'''
def sentiment(text):
    response = chat_completion(messages=[
        user("You are a sentiment classifier. For each message, give the percentage of positive/netural/negative."),
        user("I liked it"),
        assistant("70% positive 30% neutral 0% negative"),
        user("It could be better"),
        assistant("0% positive 50% neutral 50% negative"),
        user("It's fine"),
        assistant("25% positive 50% neutral 25% negative"),
        user(text),
    ])
    return response

def print_sentiment(text):
    print(f'INPUT: {text}')
    print(sentiment(text))

print_sentiment("I thought it was okay")
# More likely to return a balanced mix of positive, neutral, and negative
print_sentiment("I loved it!")
# More likely to return 100% positive
print_sentiment("Terrible service 0/10")
# More likely to return 100% negative

'''
#role prompting

#complete_and_print("Explain the pros and cons of using PyTorch.")
# More likely to explain the pros and cons of PyTorch covers general areas like documentation, the PyTorch community, and mentions a steep learning curve

#complete_and_print("Your role is a machine learning expert who gives highly technical advice to senior engineers who work with complicated datasets. Explain the pros and cons of using PyTorch.")
# Often results in more technical benefits and drawbacks that provide more technical details on how model layers

#Chain-of-Thought
#complete_and_print("Who lived longer Elvis Presley or Mozart?")
# Often gives incorrect answer of "Mozart"

#complete_and_print("Who lived longer Elvis Presley or Mozart? Let's think through this carefully, step by step.")
# Gives the correct answer "Elvis"

#RAG

MENLO_PARK_TEMPS = {
    "2023-12-11": "52 degrees Fahrenheit",
    "2023-12-12": "51 degrees Fahrenheit",
    "2023-12-13": "51 degrees Fahrenheit",
}


def prompt_with_rag(retrived_info, question):
    complete_and_print(
        f"Given the following information: '{retrived_info}', respond to: '{question}'"
    )


def ask_for_temperature(day):
    temp_on_day = MENLO_PARK_TEMPS.get(day) or "unknown temperature"
    prompt_with_rag(
        f"The temperature in Menlo Park was {temp_on_day} on {day}'",  # Retrieved fact
        f"What is the temperature in Menlo Park on {day}?",  # User question
    )

ask_for_temperature("2023-12-12")
# "Sure! The temperature in Menlo Park on 2023-12-12 was 51 degrees Fahrenheit."

ask_for_temperature("2023-07-18")

