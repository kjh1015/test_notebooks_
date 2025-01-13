import pyautogui
import keyboard
import time
from datetime import datetime
import os
import logging
import json
from PIL import ImageGrab

class WebAppPromptTester:
    def __init__(self):
        """Initialize the prompt testing automation"""
        # Setup directories and logging
        self.setup_environment()
        
        # Configure pyautogui settings
        pyautogui.FAILSAFE = True  # Move mouse to upper-left to abort
        pyautogui.PAUSE = 0.5      # Add small delay between actions
        
        # Coordinates will be set during calibration
        self.coords = {}

    def setup_environment(self):
        """Setup directories and logging"""
        self.results_dir = f"prompt_testing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logging.basicConfig(
            filename=f'{self.results_dir}/testing.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def calibrate_coordinates(self):
        """Interactive coordinate calibration"""
        print("\nCalibration Mode - Move mouse and press keys to set positions:")
        print("1. Next prompt button - press 'n'")
        print("2. Prompt edit area - press 'p'")
        print("3. Test input area - press 't'")
        print("4. Done calibrating - press 'c'")
        print("Press 'esc' to abort")
        
        while True:
            if keyboard.is_pressed('n'):
                x, y = pyautogui.position()
                self.coords['next_prompt'] = (x, y)
                print(f"Next prompt button position: {x}, {y}")
                time.sleep(0.5)
            
            elif keyboard.is_pressed('p'):
                x, y = pyautogui.position()
                self.coords['prompt_edit'] = (x, y)
                print(f"Prompt edit position: {x}, {y}")
                time.sleep(0.5)
            
            elif keyboard.is_pressed('t'):
                x, y = pyautogui.position()
                self.coords['test_input'] = (x, y)
                print(f"Test input position: {x}, {y}")
                time.sleep(0.5)
            
            elif keyboard.is_pressed('c'):
                if len(self.coords) >= 3:
                    print("Calibration complete!")
                    break
                else:
                    print("Please set all required positions first!")
                time.sleep(0.5)
            
            elif keyboard.is_pressed('esc'):
                raise Exception("Calibration aborted by user")

    def capture_screen(self, prompt_num, test_num):
        """Capture the current screen"""
        try:
            screenshot = ImageGrab.grab()
            screenshot_path = f"{self.results_dir}/prompt_{prompt_num}_test_{test_num}.png"
            screenshot.save(screenshot_path)
            return screenshot_path
        except Exception as e:
            self.logger.error(f"Failed to capture screen: {str(e)}")
            return None

    def edit_prompt(self, new_prompt):
        """Edit the prompt text"""
        try:
            # Click prompt edit area
            pyautogui.click(self.coords['prompt_edit'])
            time.sleep(0.5)
            
            # Select all existing text and delete
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.press('delete')
            time.sleep(0.5)
            
            # Type new prompt
            pyautogui.write(new_prompt)
            pyautogui.press('enter')
            time.sleep(2)  # Wait for update
            
        except Exception as e:
            self.logger.error(f"Failed to edit prompt: {str(e)}")
            raise

    def run_test_case(self, test_case, prompt_num, test_num):
        """Run a single test case"""
        try:
            # Click test input area
            pyautogui.click(self.coords['test_input'])
            time.sleep(0.5)
            
            # Clear existing text
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.press('delete')
            time.sleep(0.5)
            
            # Type test case
            pyautogui.write(test_case)
            pyautogui.press('enter')
            
            # Wait for response
            time.sleep(5)  # Adjust based on typical response time
            
            # Capture screenshot
            screenshot_path = self.capture_screen(prompt_num, test_num)
            
            return {
                'test_case': test_case,
                'screenshot': screenshot_path,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in test case: {str(e)}")
            return {'error': str(e)}

    def next_prompt(self):
        """Move to next prompt"""
        try:
            pyautogui.click(self.coords['next_prompt'])
            time.sleep(2)  # Wait for next prompt to load
            return True
        except Exception as e:
            self.logger.error(f"Failed to move to next prompt: {str(e)}")
            return False

    def generate_test_cases(self, num_cases=3):
        """Generate test cases for current prompt"""
        # Customize this based on your testing needs
        return [
            "Test case 1: Basic functionality test",
            "Test case 2: Edge case test",
            "Test case 3: Complex scenario test"
        ]

    def run_testing_session(self, num_prompts=None):
        """Run complete testing session"""
        results = {}
        prompt_count = 0
        
        try:
            while num_prompts is None or prompt_count < num_prompts:
                prompt_count += 1
                self.logger.info(f"Testing prompt #{prompt_count}")
                
                # Run test cases for current prompt
                test_results = []
                for i, test_case in enumerate(self.generate_test_cases()):
                    result = self.run_test_case(test_case, prompt_count, i)
                    test_results.append(result)
                
                # Store results
                results[f"prompt_{prompt_count}"] = {
                    'test_results': test_results
                }
                
                # Save intermediate results
                self.save_results(results)
                
                # Check for manual abort
                if keyboard.is_pressed('esc'):
                    print("\nTesting aborted by user")
                    break
                
                # Move to next prompt
                if not self.next_prompt():
                    print("\nReached end of prompts")
                    break
                
        except Exception as e:
            self.logger.error(f"Testing session failed: {str(e)}")
            print(f"\nError during testing: {str(e)}")
        
        return results

    def save_results(self, results):
        """Save results to JSON file"""
        with open(f"{self.results_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)

def main():
    try:
        print("Web App Prompt Testing Automation")
        print("--------------------------------")
        print("1. Make sure you're on the prompts page")
        print("2. We'll calibrate mouse positions first")
        print("3. Press 'esc' at any time to abort")
        input("Press Enter to start...")
        
        tester = WebAppPromptTester()
        
        # Calibrate positions
        tester.calibrate_coordinates()
        
        # Ask how many prompts to test
        num_prompts = input("\nHow many prompts to test? (Enter for all): ").strip()
        num_prompts = int(num_prompts) if num_prompts else None
        
        # Run testing session
        results = tester.run_testing_session(num_prompts)
        
        print(f"\nTesting completed. Results saved in: {tester.results_dir}")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")

if __name__ == "__main__":
    main()
    