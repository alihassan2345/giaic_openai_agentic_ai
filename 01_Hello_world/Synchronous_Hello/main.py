from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunConfig
import os
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client
)

config = RunConfig(
    model = model,
    tracing_disabled = True,
    model_provider = external_client
)

agent = Agent(
    name = "frontend_developer",
    instructions = "you are a frontend developer",
    model = model
)

question = "what is the capital of france?"

result = Runner.run_sync(agent , question , run_config = config)

print(result.final_output)