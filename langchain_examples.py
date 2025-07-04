# pip install openai
# pip install langchain
# pip install -U langchain-openai
# pip install -U langchain-community

from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
# from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableSequence
from langchain.chains import RetrievalQA


client = OpenAI()
llm_model = "gpt-4"






exit(0)  # Exit early for testing purposes









def get_completion(prompt, model=llm_model):
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.0,
    )
    return response.choices[0].message.content

print( get_completion("What is 1+1?") )


customer_email = """
    Arrr, I be fuming that me blender lid \
    flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse,\
    the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help \
    right now, matey!
"""

style = """
    American English \
    in a calm and respectful tone
"""

prompt = f"""
    Translate the text \
    that is delimited by triple backticks 
    into a style that is {style}.
    text: ```{customer_email}```
"""

print(prompt)
# print( get_completion(prompt) )



# Doing the same simple promting with LangChain

chat = ChatOpenAI(
    temperature=0.0,
    model_name=llm_model
)
print(chat)


# Prompt templating 
prompt_template = ChatPromptTemplate.from_template(
    """
    Translate the text that is delimited by triple backticks
    into a style that is {style}.
    text: ```{text}```
"""
)
print( prompt_template.messages[0].prompt )
customer_style = "American English in a calm and respectful tone"
customer_email = """
    Arrr, I be fuming that me blender lid
    flew off and splattered me kitchen walls
    with smoothie! And to make matters worse,
    the warranty don't cover the cost of
    cleaning up me kitchen. I need yer help
    right now, matey!"""
customer_message = prompt_template.format_messages(
    style = customer_style,
    text = customer_email
)
print(customer_message)
print(customer_message[0])

# Using the prompt template with the chat model
customer_response = chat.invoke(customer_message)
print(customer_response.content)


service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

service_messages = prompt_template.format_messages(
    style = service_style_pirate,
    text = service_reply
)
print(service_messages[0].content)
service_response = chat.invoke(service_messages)
print(service_response.content)


# Output Parsers
customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

prompt_template = ChatPromptTemplate.from_template(
    review_template
)
print(prompt_template)

messages = prompt_template.format_messages(
    text=customer_review
)
chat = ChatOpenAI(
    temperature=0.0,
    model_name=llm_model
)
response = chat.invoke(messages)
print(response.content)

print( type(response.content) )

# Parse LLM output string into a python dictionary
gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")
response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)



review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""
prompt = ChatPromptTemplate.from_template(review_template_2)
messages = prompt.format_messages(
    text=customer_review,
    format_instructions=format_instructions
)
print(messages[0].content)
response = chat.invoke(messages)
print(response.content)
output_dict = output_parser.parse(response.content)
print(output_dict)
print(type(output_dict))
print( output_dict.get("delivery_days"))






# Conversation Buffer Memory
chat = ChatOpenAI(
    temperature=0.0,
    model_name=llm_model
)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)
conversation.predict(input="Hi, My name is Waqas")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
print(memory.buffer)
print("printing memory variables")
print(memory.load_memory_variables({})) # Load memory variables

memory.save_context(
    {"input": "Not much, just handing"},
    {"output": "That's great to hear!"}
)
print(memory.load_memory_variables({})) # Load memory variables again

# Using ConversationBufferWindowMemory

window_memory = ConversationBufferWindowMemory(k=2)
window_memory_conversatoin = ConversationChain(
    llm=chat,
    memory=window_memory,
    verbose=True
)
window_memory_conversatoin.predict(input="Hi, My name is Malik")
window_memory_conversatoin.predict(input="What is 1+1?")
window_memory_conversatoin.predict(input="How do you explain the concept of memory in AI in one line?")
window_memory_conversatoin.predict(input="What is my name?")
window_memory_conversatoin.predict(input="What is 21+1?")


# Using ConversationTokenBufferMemory
chat = ChatOpenAI(
    temperature=0.0,
    model_name=llm_model
)
token_memory = ConversationTokenBufferMemory(
    llm=chat,
    max_token_limit=10,  # Set a token limit for the memory
)
conversation_with_token_memory = ConversationChain(
    llm=chat,
    memory=token_memory,
    verbose=True
)
token_memory.save_context(
    {"input": "Hi, My name is Waqas"},
    {"output": "Hello Waqas, nice to meet you!"}
)
token_memory.save_context(
    {"input": "What is 1+1?"},
    {"output": "1+1 equals 2."}
)
token_memory.save_context(
    {"input": "Chatbots are what?"},
    {"output": "Awesome."}
)
print( token_memory.load_memory_variables({}) )  # Load memory variables
# Note: The token memory will only keep the most recent interactions
# within the specified token limit, so older interactions may be discarded.
# This is useful for keeping the context relevant without overwhelming the model with too much history.



# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."
summary_memory = ConversationSummaryBufferMemory(
    llm=chat,
    max_token_limit=100,  # Set a token limit for the summary
)
summary_memory.save_context(
    {"input": "Hello"},
    {"output": "What's up?"}
)
summary_memory.save_context(
    {"input": "Not much, just hanging"},
    {"output": "Cool"}
)
summary_memory.save_context(
    {"input": "What is my schedule for today?"},
    {"output": schedule}
)
print(summary_memory.load_memory_variables({}))  # Load memory variables
memory_conversation = ConversationChain(
    llm=chat,
    memory=summary_memory,
    verbose=True
)
print( memory_conversation.predict(input="What is my schedule for today?") )
print( memory_conversation.predict(input="What is my schedule for tomorrow?") )





# Chains 
# llm = ChatOpenAI(
#     temperature=0.0,
#     model_name=llm_model
# )
# prompt = ChatPromptTemplate.from_template(
#     "What is the best name to describe a company that makes "
# )
# llm_chain = RunnableSequence(
#     prompt, llm
# ) | llm
# result = llm_chain.invoke("Queen Size sheet set")
# print(result)  # Output the result of the chain

# Sequential Chain
from langchain.chains import SequentialChain
llm = ChatOpenAI(
    temperature=0.0,
    model_name=llm_model
)
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)
first_chain = SequentialChain(
    chains=[first_prompt, llm],
    input_variables=["product"],
    output_variables=["name"],
    verbose=True
)
second_prompt = ChatPromptTemplate.from_template(
    "What is the best slogan for a company that makes {product}?"
)
second_chain = SequentialChain(
    chains=[second_prompt, llm],
    input_variables=["product"],
    output_variables=["slogan"],
    verbose=True
)