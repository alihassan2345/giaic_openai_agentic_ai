from agents import Agent , OpenAIChatCompletionsModel , AsyncOpenAI , RunConfig , Runner , function_tool
from dotenv import load_dotenv
import chainlit as cl
import os


gemini_api_key = os.getenv("GEMINI_API_KEY")


External_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = External_client
)

config = RunConfig(
    model = model,
    tracing_disabled = True,
    model_provider = External_client
)



agent = Agent(
    name= "ali",
    instructions = "you are helpfull assistant",
    model = model

)


@cl.on_chat_start
async def handle_start_chat():
    cl.user_session.set("history" ,[])
    await cl.Message(content="Hello from subhan kaladi").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({"role": "user", "content":message.content})
    result = await Runner.run(
        agent,
        input=history,
        run_config=config
    )
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()

