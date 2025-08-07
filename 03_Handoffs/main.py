from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    RunConfig,
    enable_verbose_stdout_logging,
)
import os
from dotenv import load_dotenv


enable_verbose_stdout_logging()

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
)

question = "what is html?"

result = Runner.run_sync(agent, question, run_config=config)

print(result.final_output)
