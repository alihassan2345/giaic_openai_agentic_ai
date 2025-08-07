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
    model="gemini-2.0-flash", openai_client=external_client
)

config = RunConfig(model=model, tracing_disabled=True, model_provider=external_client)


nextjs_developer = Agent(
    name="nextjs_developer", instructions="you are a nextjs developer", model=model
)


make_nextjs_tool = nextjs_developer.as_tool(
    tool_name="make_nextjs_tool",
    tool_description="make a nextjs tool",
)


agent = Agent(
    name="frontend_developer",
    instructions="you are a frontend developer",
    tools=[make_nextjs_tool],
)

question = "what is nextjs?"

result = Runner.run_sync(agent, question, run_config=config)

print(result.final_output)
