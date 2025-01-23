import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import ell
from typing import List
from enum import Enum
import json

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize ell
ell.init(store='./log', autocommit=True)

class UserState(Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    ANSWERING = "answering"
    FINISHED = "finished"

@ell.tool()
def check_readiness(user_input: str) -> str:
    """Analyzes user input to determine if they're ready to study."""
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7)
    response = llm.invoke(f"""
    Analyze if the user is ready to study based on their input: "{user_input}"
    Respond with JSON in this exact format:
    {{
        "is_ready": true/false,
        "message": "Motivational message or next steps"
    }}
    """)
    # Extract the content from AIMessage
    response_text = response.content
    try:
        # Try to parse as JSON first
        json.loads(response_text)
        return response_text  # If it's valid JSON, return as is
    except json.JSONDecodeError:
        # If not valid JSON, create our own
        return json.dumps({
            "is_ready": True if "yes" in response_text.lower() or "ready" in response_text.lower() else False,
            "message": response_text
        })


@ell.tool()
def generate_question(subject: str, difficulty: str) -> str:
    """Generates a question based on subject and difficulty."""
    return """
    {
        "question": "Question text",
        "answer": "Correct answer",
        "hints": ["Hint 1", "Hint 2", "Hint 3"]
    }
    """

@ell.tool()
def evaluate_answer(question: str, correct_answer: str, user_answer: str) -> str:
    """Evaluates user's answer and provides feedback."""
    return """
    {
        "is_correct": true/false,
        "feedback": "Feedback message",
        "hint": "Hint if needed"
    }
    """

class Question:
    def __init__(self, question: str, answer: str, hints: List[str]):
        self.question = question
        self.answer = answer
        self.hints = hints
        self.hint_count = 0

# ... existing imports and initial setup ...

# ... existing imports and initial setup ...

class AITutor:
    def __init__(self):
        self.state = UserState.NOT_READY
        self.current_question_index = 0
        self.questions = []
        self.current_question = None
        self.subject = "general"
        self.difficulty = "beginner"
        self.max_questions = 5

    @ell.complex(model="gpt-4-turbo-preview", tools=[check_readiness], temperature=0.7)
    def assess_readiness(self, message_history: List[ell.Message]) -> List[ell.Message]:
        """Returns a list of messages including the system prompt and message history"""
        return [
            ell.system("""You are an encouraging AI tutor. Analyze user's response to determine if they're ready to study.
            Use the check_readiness tool to evaluate their readiness.
            Return the result as a JSON string in this format:
            {
                "is_ready": true/false,
                "message": "Your message here"
            }"""),
        ] + message_history

    @ell.complex(model="gpt-4-turbo-preview", tools=[generate_question], temperature=0.7)
    def question_generator(self, message_history: List[ell.Message]) -> List[ell.Message]:
        return [
            ell.system(f"""Generate an appropriate {self.difficulty} level question about {self.subject}.
            Use the generate_question tool to create a question.
            Return the result directly."""),
        ] + message_history

    @ell.complex(model="gpt-4-turbo-preview", tools=[evaluate_answer], temperature=0.7)
    def answer_evaluator(self, message_history: List[ell.Message]) -> List[ell.Message]:
        return [
            ell.system("""Evaluate the user's answer:
            1. Check for correctness (considering alternative phrasings)
            2. Provide constructive feedback
            3. Give appropriate hint if wrong
            Return as JSON object."""),
            ell.user(message_history)
        ]

    def load_questions(self):
        """Generate questions using LLM"""
        for _ in range(self.max_questions):
            role, content_list = self.question_generator([ell.user("Generate new question")])
            try:
                if content_list and len(content_list) > 0:
                    content_block = content_list[0]
                    if hasattr(content_block, 'tool_call'):
                        result = content_block.tool_call.tool("Generate new question")
                        question_data = json.loads(result)
                        self.questions.append(Question(
                            question_data["question"],
                            question_data["answer"],
                            question_data["hints"]
                        ))
            except Exception as e:
                print(f"Debug - Question generation error: {str(e)}")

    def converse(self, user_input: str) -> str:
        """Main conversation function that handles the interaction"""
        messages = [ell.user(user_input)]
        print(f"Debug - Messages sent: {messages}")
        role_tuple, content_tuple = self.assess_readiness(messages)
        print(f"Debug - Role tuple: {role_tuple}")
        print(f"Debug - Content tuple: {content_tuple}")
        
        # Extract actual content list from the tuple
        _, content_list = content_tuple
        
        if content_list and len(content_list) > 0:
            content_block = content_list[0]
            print(f"Debug - Content block: {content_block}")
            if hasattr(content_block, 'tool_call'):
                print(f"Debug - Tool call: {content_block.tool_call}")
                # Execute the tool call and return its result directly
                result = content_block.tool_call.tool(user_input)
                print(f"Debug - Tool result: {result}")
                return result
            
        return "I couldn't understand that response."

    def check_user_readiness(self, user_input: str) -> str:
        try:
            print(f"Debug - Input: {user_input}")
            response = self.converse(user_input)
            print(f"Debug - Response from converse: {response}")
            
            # Parse the outer JSON
            outer_json = json.loads(response)
            print(f"Debug - Outer JSON: {outer_json}")
            
            # Extract and parse the inner JSON from the message field
            inner_json_str = outer_json["message"].strip('`json\n')
            inner_json = json.loads(inner_json_str)
            print(f"Debug - Inner JSON: {inner_json}")
            
            if inner_json["is_ready"]:
                if not self.questions:
                    try:
                        self.load_questions()
                        if not self.questions:  # 질문 로딩 실패
                            return "I'm having trouble generating questions. Please try again."
                    except Exception as e:
                        print(f"Debug - Question loading error: {str(e)}")
                        return "I'm having trouble generating questions. Please try again."
                
                # 질문이 성공적으로 로드된 경우에만 상태 변경
                self.state = UserState.READY
                self.current_question = self.questions[self.current_question_index]
                return f"{inner_json['message']}\n{self.get_current_question()}"
            
            return inner_json["message"]
            
        except Exception as e:
            print(f"Debug - Error: {str(e)}")
            print(f"Debug - Error type: {type(e)}")
            return "I couldn't understand that response. Are you ready to start learning?"

    def get_current_question(self) -> str:
        if self.current_question_index >= len(self.questions):
            self.state = UserState.FINISHED
            return "Congratulations! You've completed all questions!"
        
        self.current_question = self.questions[self.current_question_index]
        return self.current_question.question

    def check_answer(self, user_answer: str) -> str:
        messages = self.answer_evaluator(json.dumps({
            "question": self.current_question.question,
            "correct_answer": self.current_question.answer,
            "user_answer": user_answer
        }))
        
        try:
            response_content = [msg for msg in messages if msg.role == "assistant"][-1].content
            result = json.loads(response_content)
            
            if result["is_correct"]:
                self.current_question_index += 1
                if self.current_question_index >= len(self.questions):
                    self.state = UserState.FINISHED
                    return "Congratulations! You've completed all questions!"
                return f"{result['feedback']}\nNext question: {self.get_current_question()}"
            
            self.current_question.hint_count += 1
            if self.current_question.hint_count >= 2:
                self.current_question_index += 1
                return f"{result['feedback']}\nLet's move on. Next question: {self.get_current_question()}"
            
            return f"{result['feedback']}\nHint: {result['hint']}"
        except (json.JSONDecodeError, IndexError):
            return "I couldn't understand that response. Please try again."

    def interact(self, user_input: str) -> str:
        if self.state == UserState.FINISHED:
            return "Session completed! Type 'restart' to begin a new session."

        if user_input.lower() == "restart":
            self.__init__()
            return "Welcome to AI Tutor! Are you ready to start learning?"

        if self.state == UserState.NOT_READY:
            return self.check_user_readiness(user_input)

        # Make sure we have a current question before checking answer
        if self.state in [UserState.READY, UserState.ANSWERING] and self.current_question:
            return self.check_answer(user_input)
        else:
            # If somehow we got here without a current question, reset state
            self.state = UserState.NOT_READY
            return "Let's start over. Are you ready to begin?"

def main():
    tutor = AITutor()
    print("Welcome to AI Tutor! Are you ready to start learning?")
    
    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            print("Thanks for learning with AI Tutor!")
            break
            
        response = tutor.interact(user_input)
        print(response)

if __name__ == "__main__":
    main()
    