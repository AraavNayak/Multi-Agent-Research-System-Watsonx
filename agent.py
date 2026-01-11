from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool # giving agents access to tools
from langchain_ibm import WatsonxLLM # interface for WatsonX
import os #api keys for environment

os.environ['WATSONX_APIKEY'] = ""
os.environ['SERPER_API_KEY'] = ""

# Parameters
parameters = {"decoding_method": "greedy", "max_new_tokens": 500}

# Create first LLM
llm = WatsonxLLM(
    model_id='meta-llama/llama-3-70b-instruct',
    url="https://us-south.ml.cloud.ibm.com",
    params = parameters,
    project_id = "c5b4b29a-0fcb-462e-a0a3-51a00d725885"
)

# function calling llm
function_calling_llm = WatsonxLLM(
    model_id='ibm-mistralai/merlinite-7b',
    url="https://us-south.ml.cloud.ibm.com",
    params = parameters,
    project_id = "c5b4b29a-0fcb-462e-a0a3-51a00d725885"
)

# Tools
search = SerperDevTool()


# Create a research agent
researcher = Agent(
    llm=llm, function_calling_llm=function_calling_llm, role="Senior AI Researcher", 
    goal="Find promising research in the field of quantum computing", 
    backstory="You are a veteran quantum computing researcher with a pHD in modern physics",
    allow_delegation=False, tools=[search], verbose=1
)

# delegation allows agents to hand abilities to each other and coordinate amongst each other
# Create a task
task1 = Task(
    description="Search the internet and find five promising examples of AI research",
    expected_output="A detailed bullet point summary on each of the topics. Each bullet point should  cover topic and background, and why the innovation is useful",
    output_file="task1output.txt",
    agent=researcher
)

# Create the second agent
writer = Agent(
    llm=llm, role="Senior Speech Writer", 
    goal="Write engaging and witty keynote speeches from provided research", 
    backstory="You are a veteran quantum computing writer with a background in modern physics",
    allow_delegation=False, verbose=1
)

# Second task
task12= Task(
    description="Write an engaging keynote speech on quantum computing",
    expected_output="A detailed keynote speech with an introduction, body, and conclusion",
    output_file="task2output.txt",
    agent=writer
)

# Put it all together with the crew
crew = Crew(agents=[researcher, writer], tasks=[task1], verbose=1)
print(crew.kickoff())

