from typing import Dict, List, Any, TypedDict, Annotated, Sequence, Union
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import pyautogui
import time
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
import operator

# Configure PyAutoGUI safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

# Define custom types for state management
class AgentState(TypedDict):
    messages: List[str]
    current_action: str
    screen_info: Dict[str, Any]
    action_history: List[str]
    next_action: Dict[str, str]

# Helper functions for computer control
def safe_click(x: int, y: int):
    """Safely move to and click at coordinates"""
    pyautogui.moveTo(x, y, duration=0.5)
    pyautogui.click()

def type_text(text: str):
    """Type text with natural pauses"""
    pyautogui.typewrite(text, interval=0.1)

def find_and_click(image_path: str) -> bool:
    """Find an image on screen and click it"""
    try:
        location = pyautogui.locateCenterOnScreen(image_path)
        if location:
            safe_click(location.x, location.y)
            return True
        return False
    except Exception as e:
        print(f"Error finding image: {e}")
        return False

# Tool definitions
def mouse_click_tool(coordinates: str) -> str:
    """Click at specified coordinates (x,y)"""
    try:
        x, y = map(int, coordinates.split(','))
        safe_click(x, y)
        return f"Clicked at coordinates ({x}, {y})"
    except Exception as e:
        return f"Error clicking: {str(e)}"

def type_text_tool(text: str) -> str:
    """Type the specified text"""
    try:
        type_text(text)
        return f"Typed text: {text}"
    except Exception as e:
        return f"Error typing text: {str(e)}"

def screenshot_tool() -> str:
    """Take a screenshot of the current screen"""
    try:
        screenshot = pyautogui.screenshot()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        screenshot.save(filename)
        return f"Screenshot saved as {filename}"
    except Exception as e:
        return f"Error taking screenshot: {str(e)}"

def find_image_tool(image_path: str) -> str:
    """Find and click on an image on screen"""
    if find_and_click(image_path):
        return f"Found and clicked image: {image_path}"
    return f"Could not find image: {image_path}"

# Create tools list
tools = [
    Tool.from_function(
        func=mouse_click_tool,
        name="mouse_click",
        description="Click at specified coordinates (x,y). Input should be 'x,y'"
    ),
    Tool.from_function(
        func=type_text_tool,
        name="type_text",
        description="Type the specified text"
    ),
    Tool.from_function(
        func=screenshot_tool,
        name="take_screenshot",
        description="Take a screenshot of the current screen"
    ),
    Tool.from_function(
        func=find_image_tool,
        name="find_and_click_image",
        description="Find and click on an image on screen. Input should be path to image file"
    )
]

# Initialize tool node
tools_node = ToolNode(tools)

# Create LLM for decision making
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

def analyze_state(state: Dict) -> Dict:
    """Analyze current state and decide next action"""
    try:
        # Generate next action using LLM
        messages = [
            HumanMessage(content=f"""Current screen state: {state['screen_info']}
            Action history: {state['action_history']}
            User request: {state['messages'][-1] if state['messages'] else 'No messages'}
            
            What action should be taken next? Available tools:
            - mouse_click: Click at coordinates (x,y)
            - type_text: Type specified text
            - take_screenshot: Take a screenshot
            - find_and_click_image: Find and click image on screen
            
            Respond with the tool name and input in format: tool_name|||input
            
            Example responses:
            mouse_click|||100,200
            type_text|||Hello World
            take_screenshot|||
            find_and_click_image|||button.png""")
        ]
        response = llm.invoke(messages).content
        
        if '|||' not in response:
            raise ValueError(f"Invalid response format: {response}")
            
        tool_name, tool_input = response.split("|||", 1)
        state['next_action'] = {
            "tool": tool_name.strip(),
            "tool_input": tool_input.strip()
        }
        return state
        
    except Exception as e:
        print(f"Error in analyze_state: {str(e)}")
        state['next_action'] = {
            "tool": "take_screenshot",
            "tool_input": ""
        }
        return state

def execute_tools(state: Dict) -> Dict:
    """Execute the selected action using tools"""
    try:
        # Format the input as expected by ToolNode
        tool_message = AIMessage(content=f"{{\"name\": \"{state['next_action']['tool']}\", \"arguments\": \"{state['next_action']['tool_input']}\"}}") 
        
        result = tools_node.invoke({"messages": [tool_message]})
        state['current_action'] = f"{state['next_action']['tool']}: {result}"
        state['action_history'].append(state['current_action'])
        return state
    except Exception as e:
        print(f"Error executing tool: {e}")
        state['current_action'] = f"ERROR: {str(e)}"
        state['action_history'].append(state['current_action'])
        return state

def should_continue(state: Dict) -> Union[Annotated[bool, "continue"], Annotated[bool, "end"]]:
    """Determine if the agent should continue or stop"""
    max_actions = 10
    if len(state['action_history']) >= max_actions:
        return "end"
    return "continue"

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze", analyze_state)
workflow.add_node("execute", execute_tools)

# Add edges
workflow.add_edge("analyze", "execute")

# Add conditional edges
workflow.add_conditional_edges(
    "execute",
    should_continue,
    {
        "continue": "analyze",
        "end": END
    }
)

# Set entry point
workflow.set_entry_point("analyze")

# Compile the graph
agent = workflow.compile()

def main():
    """Main function to run the computer control agent"""
    initial_state = AgentState(
        messages=["Take a screenshot and click at coordinates 100,100"],
        current_action="",
        screen_info={"resolution": pyautogui.size()},
        action_history=[],
        next_action={}
    )
    
    try:
        print("Starting agent with initial state:")
        print(f"Message: {initial_state['messages'][0]}")
        print(f"Screen info: {initial_state['screen_info']}\n")
        
        final_state = agent.invoke(initial_state)
        
        print("\nAction history:")
        for action in final_state["action_history"]:
            print(f"- {action}")
            
    except Exception as e:
        print(f"\nError running agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()