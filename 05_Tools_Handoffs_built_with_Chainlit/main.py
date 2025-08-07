from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    RunConfig,
    Runner,
    function_tool,
)
from dotenv import load_dotenv
import chainlit as cl
import os


gemini_api_key = os.getenv("GEMINI_API_KEY")


External_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=External_client
)

config = RunConfig(model=model, tracing_disabled=True, model_provider=External_client)


@function_tool
def weather_app(city: str) -> str:
    return f"The weather in {city} is currently winter."


@function_tool
def calculate_area(length: float, width: float) -> str:
    area = length * width
    return f"The area is {area} square units."


@function_tool
def greet_user(name: str) -> str:
    return f"Hello {name}, welcome to the system!"


@function_tool
def convert_celsius_to_fahrenheit(celsius: float) -> str:
    f = (celsius * 9 / 5) + 32
    return f"{celsius}°C is equal to {f:.2f}°F"


@function_tool
def reverse_string(text: str) -> str:
    return text[::-1]


@function_tool
def word_count(text: str) -> str:
    count = len(text.split())
    return f"Word count: {count}"


@function_tool
def is_even(number: int) -> str:
    return "Even" if number % 2 == 0 else "Odd"


@function_tool
def get_day_message(day: str) -> str:
    messages = {
        "monday": "Start strong!",
        "friday": "Weekend is near!",
        "sunday": "Relax and recharge.",
    }
    return messages.get(day.lower(), "Just another day!")


@function_tool
def bmi_calculator(weight: float, height: float) -> str:
    bmi = weight / (height**2)
    return f"Your BMI is {bmi:.2f}"


@function_tool
def is_palindrome(word: str) -> str:
    return (
        "Yes, it's a palindrome."
        if word == word[::-1]
        else "No, it's not a palindrome."
    )


@function_tool
def time_greeting(hour: int) -> str:
    if hour < 12:
        return "Good morning!"
    elif hour < 18:
        return "Good afternoon!"
    else:
        return "Good evening!"


@function_tool
def sum_of_list(numbers: list[int]) -> str:
    return f"The total sum is {sum(numbers)}"


@function_tool
def generate_email(name: str, domain: str) -> str:
    return f"{name.lower().replace(' ', '_')}@{domain.lower()}"


@function_tool
def countdown(start: int) -> str:
    return ", ".join(str(i) for i in range(start, 0, -1))


@function_tool
def currency_converter(amount: float, rate: float) -> str:
    return f"Converted amount: {amount * rate:.2f}"


@function_tool
def char_count(text: str) -> str:
    return f"Total characters: {len(text)}"


@function_tool
def get_initials(name: str) -> str:
    return "".join(part[0].upper() for part in name.split())


@function_tool
def square_number(number: int) -> str:
    return f"Square of {number} is {number ** 2}"


@function_tool
def password_strength(password: str) -> str:
    length = len(password)
    if (
        length >= 12
        and any(c.isdigit() for c in password)
        and any(c.isupper() for c in password)
    ):
        return "Strong"
    elif length >= 8:
        return "Medium"
    else:
        return "Weak"


@function_tool
def make_slug(title: str) -> str:
    return title.lower().replace(" ", "-")


html_agent = Agent(
    name="html_expert",
    instructions="You are an expert in HTML. Structure web pages using semantic HTML5 and ensure accessibility.",
    model=model,
)

css_agent = Agent(
    name="css_stylist",
    instructions="You are a CSS expert. Style pages, manage layout, flexbox, grid, and responsiveness.",
    model=model,
)

tailwind_agent = Agent(
    name="tailwind_specialist",
    instructions="You are a Tailwind CSS expert. Use utility classes to convert design into responsive UI.",
    model=model,
)

js_agent = Agent(
    name="js_engineer",
    instructions="You are a JavaScript expert. Add interactivity using vanilla JS with best practices.",
    model=model,
)

ts_agent = Agent(
    name="typescript_developer",
    instructions="You are a TypeScript specialist. Use strong typing for variables, functions, and API responses.",
    model=model,
)

react_agent = Agent(
    name="react_developer",
    instructions="You are a React developer. Build UI components with hooks, props, and state management.",
    model=model,
)

nextjs_agent = Agent(
    name="nextjs_engineer",
    instructions="You are a Next.js developer. Handle SSR, routing, and API endpoints efficiently.",
    model=model,
)

uiux_agent = Agent(
    name="uiux_designer",
    instructions="You are a UI/UX designer. Focus on usability, layout hierarchy, and visual design.",
    model=model,
)

a11y_agent = Agent(
    name="accessibility_auditor",
    instructions="You are an accessibility expert. Make sure UIs follow WCAG and ARIA standards.",
    model=model,
)

perf_agent = Agent(
    name="performance_optimizer",
    instructions="You are a performance specialist. Optimize web apps for speed, Lighthouse score, and Core Web Vitals.",
    model=model,
)


agent = Agent(
    name="frontend_developer",
    instructions="you are a frontend developer",
    handoffs=[
        html_agent,
        css_agent,
        tailwind_agent,
        js_agent,
        ts_agent,
        react_agent,
        nextjs_agent,
        uiux_agent,
        a11y_agent,
        perf_agent,
    ],
    tools=[
        weather_app,
        calculate_area,
        greet_user,
        convert_celsius_to_fahrenheit,
        reverse_string,
        word_count,
        is_even,
        get_day_message,
        bmi_calculator,
        is_palindrome,
        time_greeting,
        sum_of_list,
        generate_email,
        countdown,
        currency_converter,
        char_count,
        get_initials,
        square_number,
        password_strength,
        make_slug,
    ],
)


@cl.on_chat_start
async def handle_start_chat():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello from ali hassan").send()


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({"role": "user", "content": message.content})
    result = await Runner.run(agent, input=history, run_config=config)
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()
