
import pytest
import logging
from crewai import Agent, Task, Crew, Process

# Set up logging
logging.basicConfig(level=logging.DEBUG)
# Define your agents
language_agent = Agent(
    role='Language Selector',
    goal='Determine the language for the greeting.',
    backstory="You are an expert in language selection, choosing the most appropriate language based on context.",
    allow_delegation=False,
    llm="gpt-4o-mini" # or "openai/gpt-4o-mini"
)

greeting_agent = Agent(
    role='Simple Greeter',
    goal='Say hello to the world in a specified language.',
    backstory="You are a friendly agent who greets the world in various languages.",
    allow_delegation=False,
    llm="gpt-4o-mini" # or "openai/gpt-4o-mini"
)

# Define your tasks
language_task = Task(
    description='Decide which language the greeting should be in. Choose from English, Spanish, French, or German.',
    agent=language_agent,
    expected_output="A string representing a language: 'English', 'Spanish', 'French', or 'German'."
)

greeting_task = Task(
    description="Generate a simple greeting in the language specified by the Language Selector agent.",
    agent=greeting_agent,
    expected_output="A greeting message in the selected language.",
    context=[language_task] # Make the output of language_task available to this task
)

# Instantiate your crew
crew = Crew(
    agents=[language_agent, greeting_agent],
    tasks=[language_task, greeting_task],
    process=Process.sequential,
)

@pytest.mark.asyncio
def testCrewai():
    # Kick off the crew
    result = crew.kickoff()
    print(result)