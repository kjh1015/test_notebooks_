from typing import Annotated, Sequence, TypedDict, Union, cast
from langgraph.graph import Graph, MessageGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

# Define our state types
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    tasks: Annotated[list[str], "List of tasks to be executed"]
    current_task: Annotated[Union[str, None], "Current task being processed"]
    task_plans: Annotated[dict, "Plans for each task"]
    results: Annotated[dict, "Results from executed tasks"]
    final_report: Annotated[Union[str, None], "Final compilation of results"]

# Initialize our LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define mock tools
class MockTools:
    @staticmethod
    def web_search(query: str) -> str:
        return f"Mock web search results for: {query}"
    
    @staticmethod
    def analyze_data(data: str) -> str:
        return f"Mock analysis results for: {data}"
    
    @staticmethod
    def generate_recommendations(analysis: str) -> str:
        return f"Mock recommendations based on: {analysis}"

# Initialize tools
tools = MockTools()

# Task Division Node
task_division_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", """Break down the user's request into specific, actionable tasks. 
    Each task should be clear, concrete, and executable.
    
    For example, if the task is about sleeping well, you might break it down into:
    [
        "Analyze current sleep patterns and issues",
        "Identify optimal bedtime routine",
        "Recommend environmental improvements for sleep",
        "Suggest dietary and lifestyle adjustments for better sleep",
        "Create a sleep improvement implementation plan"
    ]
    
    Return ONLY the Python list of tasks, formatted exactly like the example above.
    Do not include any additional text or explanations.""")
])

def divide_tasks(state: AgentState) -> AgentState:
    """Divide the main task into subtasks"""
    print("\n=== Dividing Tasks ===")
    # Initialize state variables
    if "tasks" not in state:
        state["tasks"] = []
    if "results" not in state:
        state["results"] = {}
    if "task_plans" not in state:
        state["task_plans"] = {}
    if "current_task" not in state:
        state["current_task"] = None
    
    print("Input message:", state["messages"][0].content)
    response = llm.invoke(
        task_division_prompt.format_messages(messages=state["messages"])
    )
    print("LLM Response:", response.content)
    
    # Safer parsing of the response
    try:
        content = response.content.strip()
        if content.startswith('[') and content.endswith(']'):
            import ast
            tasks = ast.literal_eval(content)
        else:
            tasks = [line.strip() for line in content.split('\n') 
                    if line.strip() and not line.startswith('#')]
        
        if not tasks:
            print("No tasks found, creating default tasks...")
            tasks = [
                "Analyze current sleep patterns and issues",
                "Identify optimal bedtime routine",
                "Recommend environmental improvements for sleep",
                "Suggest dietary and lifestyle adjustments",
                "Create a sleep improvement implementation plan"
            ]
        
        state["tasks"] = tasks
        state["current_task"] = tasks[0]
        print("Generated Tasks:", tasks)
        print("Current Task:", state["current_task"])
        
    except Exception as e:
        print(f"Error in task division: {e}")
        tasks = [
            "Analyze sleep patterns",
            "Create sleep improvement plan",
            "Generate recommendations"
        ]
        state["tasks"] = tasks
        state["current_task"] = tasks[0]
        print("Using fallback tasks:", tasks)
    
    return state

# Task Planning Node
planning_prompt = ChatPromptTemplate.from_messages([
    ("system", """For the given task, create a detailed plan using these available tools:
    - web_search: Search for relevant information
    - analyze_data: Analyze provided data
    - generate_recommendations: Generate specific recommendations

    Return your plan as a valid Python dictionary in this exact format:
    {
        'tools': ['web_search', 'analyze_data', 'generate_recommendations'],
        'steps': [
            'Search for sleep research using web_search',
            'Analyze sleep patterns using analyze_data',
            'Generate sleep recommendations using generate_recommendations'
        ],
        'expected_output': 'Detailed sleep improvement plan with specific recommendations'
    }"""),
    ("user", "Create a detailed plan for this task: {current_task}")
])

def plan_task(state: AgentState | str) -> AgentState:
    """Plan the next task"""
    print("\n=== Planning Task ===")
    # If we got a string, initialize a new state
    if isinstance(state, str):
        print(f"Received string instead of state: {state}")
        return {
            "tasks": [],
            "current_task": None,
            "task_plans": {},
            "results": {},
            "final_report": None,
            "messages": []
        }
    
    # Initialize state variables
    if "tasks" not in state:
        state["tasks"] = []
    if "results" not in state:
        state["results"] = {}
    if "task_plans" not in state:
        state["task_plans"] = {}
    if "current_task" not in state:
        state["current_task"] = None
    
    print("Current Task:", state.get("current_task"))
    print("All Tasks:", state.get("tasks", []))
    
    try:
        response = llm.invoke(
            planning_prompt.format_messages(current_task=state["current_task"])
        )
        print("Plan Response:", response.content)
        
        content = response.content.strip()
        if '{' in content and '}' in content:
            dict_start = content.find('{')
            dict_end = content.rfind('}') + 1
            dict_str = content[dict_start:dict_end]
            
            import ast
            plan = ast.literal_eval(dict_str)
        else:
            plan = {
                'tools': ['analysis'],
                'steps': [content],
                'expected_output': 'Analysis and recommendations'
            }
        
        state["task_plans"][state["current_task"]] = plan
        print("Created Plan:", plan)
        
    except Exception as e:
        print(f"Error in planning: {e}")
        plan = {
            'tools': ['analysis'],
            'steps': [f"Analyze and execute: {state['current_task']}"],
            'expected_output': 'Analysis results and recommendations'
        }
        state["task_plans"][state["current_task"]] = plan
        print("Using fallback plan:", plan)
    
    return state

# Task Execution Node
execution_prompt = ChatPromptTemplate.from_messages([
    ("system", """Execute the task using the available tools:
    - web_search: Search for relevant information
    - analyze_data: Analyze provided data
    - generate_recommendations: Generate specific recommendations
    
    Follow the plan and use the appropriate tools for each step."""),
    ("user", """Task: {current_task}
Plan: {current_plan}

Return your response as a valid Python dictionary with status and details keys.
Example: {{'status': 'completed', 'details': 'Based on web search and analysis...'}}""")
])

def execute_task(state: AgentState | str) -> AgentState:
    """Execute the current task using available tools"""
    if isinstance(state, str):
        state = {
            "tasks": [],
            "current_task": None,
            "task_plans": {},
            "results": {},
            "final_report": None,
            "messages": []
        }
    
    if "results" not in state:
        state["results"] = {}
    if "task_plans" not in state:
        state["task_plans"] = {}
    
    task = state["current_task"]
    plan = state["task_plans"][task]
    
    # Execute each step using appropriate tools
    try:
        steps_results = []
        for step in plan['steps']:
            if 'web_search' in step:
                result = tools.web_search(task)
            elif 'analyze_data' in step:
                result = tools.analyze_data(task)
            elif 'generate_recommendations' in step:
                result = tools.generate_recommendations(task)
            else:
                result = f"Executed step: {step}"
            steps_results.append(result)
        
        results = {
            'status': 'completed',
            'details': '\n'.join(steps_results)
        }
    except Exception as e:
        results = {
            'status': 'error',
            'details': str(e)
        }
    
    state["results"][task] = results
    return state

# Report Generation Node
report_prompt = ChatPromptTemplate.from_messages([
    ("system", """Create a comprehensive report combining all task results.
    The report should be clear and well-structured.
    Include any relevant insights or recommendations."""),
    ("user", "Results: {results}")
])

def generate_report(state: AgentState | str) -> AgentState:
    """Generate final report"""
    # If we got a string, initialize a new state
    if isinstance(state, str):
        state = {
            "tasks": [],
            "current_task": None,
            "task_plans": {},
            "results": {},
            "final_report": None,
            "messages": []
        }
    
    # Initialize state variables
    if "results" not in state:
        state["results"] = {}
    if "final_report" not in state:
        state["final_report"] = None
    
    response = llm.invoke(
        report_prompt.format_messages(results=state["results"])
    )
    state["final_report"] = response.content
    return state

# Task Management Node
def should_continue(state: AgentState | str) -> dict:
    """Function to determine the next step in the workflow"""
    print("\n=== Checking Next Step ===")
    # If we got a string, return end
    if isinstance(state, str):
        print(f"Received string state: {state}")
        return {"next": "end"}
    
    print("Current Task:", state.get("current_task"))
    print("All Tasks:", state.get("tasks", []))
    
    # If no tasks available, end workflow
    if not state.get("tasks"):
        print("No tasks available, ending workflow")
        return {"next": "end"}
    
    try:
        current_idx = state["tasks"].index(state["current_task"])
        if current_idx < len(state["tasks"]) - 1:
            next_task = state["tasks"][current_idx + 1]
            print(f"Moving to next task: {next_task}")
            state["current_task"] = next_task
            return {"next": "plan", "state": state}  # Return both next step and state
        print("All tasks completed, generating report")
        return {"next": "report", "state": state}  # Return both next step and state
    except ValueError:
        print("Current task not found in task list")
        return {"next": "end"}

# Add this function to handle the end state
def end_workflow(state: AgentState) -> AgentState:
    """Function to handle the end of the workflow"""
    return state

# Build the graph
workflow = Graph()

# Add nodes
workflow.add_node("divide", divide_tasks)
workflow.add_node("plan", plan_task)
workflow.add_node("execute", execute_task)
workflow.add_node("report", generate_report)
workflow.add_node("should_continue", should_continue)
workflow.add_node("end", end_workflow)

# Add edges
workflow.add_edge("divide", "plan")
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "should_continue")
workflow.add_edge("report", "end")

# Add conditional edges with state preservation
workflow.add_conditional_edges(
    "should_continue",
    {
        "plan": lambda x: plan_task(x.get("state", x)),
        "report": lambda x: generate_report(x.get("state", x)),
        "end": lambda x: end_workflow(x.get("state", x))
    }
)

workflow.set_entry_point("divide")

# Create the runnable
chain = workflow.compile()

# Example usage
def process_query(query: str) -> dict:
    """Process the user query"""
    try:
        # Add more context to the query
        enriched_query = f"""Please help me with this task: {query}

        I need a detailed analysis and actionable plan. Consider all relevant factors and provide specific recommendations.
        Break this down into clear, actionable subtasks that we can analyze individually."""

        initial_state: AgentState = {
            "messages": [HumanMessage(content=enriched_query)],
            "tasks": [],
            "current_task": None,
            "task_plans": {},
            "results": {},
            "final_report": None
        }
        
        result = chain.invoke(initial_state)
        print("\nRaw result:", result)  # Debug print
        
        # Handle different result structures
        if isinstance(result, dict):
            if "state" in result:
                result = result["state"]
            elif "next" in result and result.get("plan"):
                result = result["plan"]
                
        # Ensure we have valid results with default values
        return {
            "tasks": result.get("tasks", []),
            "plans": result.get("task_plans", {}),
            "results": result.get("results", {}),
            "final_report": result.get("final_report", "No report generated")
        }
        
    except Exception as e:
        print(f"\nError processing query: {str(e)}")
        # Return default structure with error message
        return {
            "tasks": [],
            "plans": {},
            "results": {},
            "final_report": f"Error: {str(e)}"
        }

# Example usage with interactive input
if __name__ == "__main__":
    print("Task Analysis System")
    print("===================")
    print("Enter your task below (press Enter twice to finish):")
    
    # Collect multiline input
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        elif lines:  # Empty line and we have content
            break
    
    query = "\n".join(lines)
    
    if not query.strip():
        print("No input provided. Using example query...")
        query = """Analyze the performance of our e-commerce website and suggest improvements. 
        Consider factors like:
        - Page load times
        - User experience
        - Conversion rates
        - Mobile responsiveness
        """
    
    print("\nProcessing your request...\n")
    try:
        result = process_query(query)
        print("\n=== Tasks ===")
        for i, task in enumerate(result["tasks"], 1):
            print(f"{i}. {task}")
            
        print("\n=== Plans ===")
        for task, plan in result["plans"].items():
            print(f"\nTask: {task}")
            print("Tools:", plan.get("tools", []))
            print("Steps:", plan.get("steps", []))
            
        print("\n=== Results ===")
        for task, res in result["results"].items():
            print(f"\nTask: {task}")
            print("Status:", res.get("status"))
            print("Details:", res.get("details"))
            
        print("\n=== Final Report ===")
        print(result["final_report"])
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease try again with a more specific query.")