from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    RunConfig,
    function_tool,
)
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


@function_tool
def weather_app(city: str) -> str:
    print("weather tool call")
    return f"the weather in {city} is winter"


@function_tool
def addition(a: int, b: int) -> int:
    print("additional tool call")
    return f"{a} + {b}"


@function_tool
def substraction(a: int, b: int) -> int:
    print("substracton tool call")
    return f"{a} - {b}"


@function_tool
def multiplication(a: int, b: int) -> int:
    print("multiplication tool call")
    return f"{a} * {b}"


@function_tool
def division(a: int, b: int) -> int:
    print("division tool call")
    return f"{a} / {b}"


agent = Agent(
    name="frontend_developer",
    instructions="you are a frontend developer",
    tools=[weather_app, addition, substraction , multiplication , division],
    model=model,
)

question = input("enter your question: ")

result = Runner.run_sync(agent, question, run_config=config)

print(result.final_output)
