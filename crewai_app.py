from crewai import Agent, Task, Crew, Process,LLM

# Create agents
researcher = Agent(
    role='Research Analyst',
    goal='Conduct thorough research and analysis',
    backstory='Expert researcher with attention to detail',
    verbose=True,
    llm=LLM(model="ollama/mistral", base_url="http://localhost:11434"),
    allow_delegation=False  # Helps with memory management
)

# Define tasks - keeping it lighter for M1
research_task = Task(
    description="""Research the topic: 'Benefits of Electric Vehicles'.
    Focus on:
    1. Environmental impact
    2. Cost savings
    3. Recent technological improvements
    Provide 3-4 key points for each area.""",
    expected_output="""A comprehensive research summary detailing the benefits of Electric Vehicles, structured in three main sections:

1. Environmental Impact section containing 3-4 specific data-backed points about emissions reduction, carbon footprint, and ecological benefits.
2. Cost Savings section containing 3-4 detailed points about financial advantages, including maintenance, fuel savings, and incentives.
3. Technological Improvements section containing 3-4 recent advancements in EV technology, focusing on batteries, charging, and performance.

Each point should be clearly explained in 2-3 sentences with relevant statistics or examples where applicable.""",
    agent=researcher
)

# Create crew with minimal agents for better performance
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True,
    process=Process.sequential
)

# Execute
result = crew.kickoff()

print("\nFinal Result:")
print(result)
