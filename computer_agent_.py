from typing import Dict, List, Any, TypedDict
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
import pyautogui
import time
import os
from dotenv import load_dotenv
import base64
# Load environment variables
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# Configure PyAutoGUI safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

class ComputerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1024)
        self.screenshot_dir = "screenshots"
        self.ensure_screenshot_dir()
        
    def ensure_screenshot_dir(self):
        """Create screenshots directory if it doesn't exist"""
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def take_screenshot(self) -> str:
        """Take a screenshot and return the filename"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        return filename

    def click(self, x: int, y: int) -> str:
        """Click at specific coordinates"""
        # Get screen size
        screen_width, screen_height = pyautogui.size()
        
        print(f"Screen size: {screen_width}x{screen_height}")
        print(f"Requested coordinates: ({x}, {y})")
        
        # Ensure coordinates are within screen bounds
        x = max(0, min(x, screen_width - 1))
        y = max(0, min(y, screen_height - 1))
        
        print(f"Adjusted coordinates: ({x}, {y})")
        
        pyautogui.moveTo(x, y, duration=0.5)
        pyautogui.click()
        return f"Clicked at ({x}, {y})"

    def type_text(self, text: str) -> str:
        """Type text with natural pauses"""
        pyautogui.typewrite(text, interval=0.1)
        return f"Typed: {text}"

    def analyze_screen(self, screenshot_path: str, user_input: str) -> Dict:
        """Analyze screenshot using GPT-4V and determine next action"""
        screen_width, screen_height = pyautogui.size()
        base64_image = self.encode_image_to_base64(screenshot_path)
        
        messages = [
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": f"""User request: {user_input}
                    
                    Screen size: {screen_width}x{screen_height}
                    
                    IMPORTANT INSTRUCTIONS:
                    1. Only suggest clicking coordinates for elements you can actually see in the screenshot
                    2. Do not make assumptions about icon locations
                    3. All coordinates must be within (0,0) to ({screen_width},{screen_height})
                    4. If you can't see what you're looking for, say so and suggest taking another screenshot
                    
                    Available actions:
                    1. Click at coordinates (x,y)
                    2. Type text
                    3. Take another screenshot
                    4. Wait for a moment
                    
                    Respond in this format:
                    ANALYSIS: <Describe exactly what you see in the screenshot>
                    ACTION: <action_type>
                    PARAMS: <parameters>"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ])
        ]
        
        response = self.llm.invoke(messages).content
        print(f"\nAnalysis result:\n{response}\n")
        return self.parse_action(response)

    def parse_action(self, response: str) -> Dict:
        """Parse LLM response into action and parameters"""
        try:
            lines = response.strip().split('\n')
            action_line = next(line for line in lines if line.startswith('ACTION:'))
            params_line = next(line for line in lines if line.startswith('PARAMS:'))
            
            # Remove spaces and convert to lowercase for consistent matching
            action = action_line.replace('ACTION:', '').strip().lower().replace(' ', '')
            params = params_line.replace('PARAMS:', '').strip().lower()
            
            # Normalize action names
            if action in ['takeanotherscreenshot', 'takescreenshot']:
                action = 'screenshot'
            
            # Set default parameter for wait action if 'none' is specified
            if action == "wait" and (params == "none" or not params):
                params = "1"
            
            return {"action": action, "params": params}
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {"action": "wait", "params": "1"}

    def execute_action(self, action: Dict) -> str:
        """Execute the specified action"""
        action_type = action["action"]
        params = action["params"]
        
        if action_type == "click":
            try:
                x, y = map(int, params.split(','))
                return self.click(x, y)
            except ValueError:
                return "Error: Invalid click coordinates"
                
        elif action_type == "type":
            return self.type_text(params)
            
        elif action_type == "wait":
            try:
                seconds = float(params)
                time.sleep(seconds)
                return f"Waited for {seconds} seconds"
            except ValueError:
                return "Error: Invalid wait duration"
                
        elif action_type == "screenshot":
            return f"Took new screenshot: {self.take_screenshot()}"
            
        return f"Unknown action: {action_type}"

    def run(self):
        """Main loop for the computer agent"""
        print("Computer Control Agent (with Vision)")
        print("Enter 'quit' to exit")
        
        while True:
            # Get user input
            print("\n" + "="*50)
            user_input = input("What would you like me to do? ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            try:
                # Take initial screenshot
                print("\nTaking screenshot...")
                screenshot_path = self.take_screenshot()
                print(f"Screenshot saved: {screenshot_path}")
                
                # Analyze and act
                print("\nAnalyzing screen with GPT-4V...")
                action = self.analyze_screen(screenshot_path, user_input)
                
                # Execute action
                print("\nExecuting action...")
                result = self.execute_action(action)
                print(f"Result: {result}")
                
            except Exception as e:
                print(f"\nError: {str(e)}")

if __name__ == "__main__":
    agent = ComputerAgent()
    agent.run()
    