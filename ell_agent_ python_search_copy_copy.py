import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import ell
from typing import List, Optional, Dict
from enum import Enum
import json
import pandas as pd

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
    Consider both English and Korean responses.
    Respond with JSON in this exact format:
    {{
        "is_ready": true/false,
        "message": "Motivational message or next steps"
    }}
    """)
    response_text = response.content
    try:
        json.loads(response_text)
        return response_text
    except json.JSONDecodeError:
        return json.dumps({
            "is_ready": True if "yes" in response_text.lower() or "ready" in response_text.lower() else False,
            "message": response_text
        })

@ell.tool()
def generate_question(subject: str, difficulty: str, dataset_question: Optional[Dict] = None) -> str:
    """Generates a question based on subject and difficulty, optionally using dataset."""
    if dataset_question:
        return json.dumps({
            "question": dataset_question["질의"],
            "answer": dataset_question["응답"],
            "hints": [
                f"Think about {dataset_question['학습유형']}",
                f"Consider the context: {dataset_question['대단원']}",
                f"Focus on: {dataset_question['중단원']}"
            ]
        })
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
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7)
    response = llm.invoke(f"""
    Evaluate this answer:
    Question: {question}
    Correct Answer: {correct_answer}
    User Answer: {user_answer}
    
    Consider both Korean and English responses.
    Return JSON in this format:
    {{
        "is_correct": true/false,
        "feedback": "Feedback message",
        "hint": "Hint if needed"
    }}
    """)
    return response.content

class Question:
    def __init__(self, question: str, answer: str, hints: List[str], metadata: Optional[Dict] = None):
        self.question = question
        self.answer = answer
        self.hints = hints
        self.hint_count = 0
        self.metadata = metadata or {}

class AITutor:
    def __init__(self, dataset_path: Optional[str] = None):
        self.state = UserState.NOT_READY
        self.current_question_index = 0
        self.questions = []
        self.current_question = None
        self.subject = "general"
        self.difficulty = "beginner"
        self.max_questions = 5
        self.dataset = None
        
        if dataset_path:
            try:
                print(f"Attempting to load dataset from: {dataset_path}")  # Debug print
                if not os.path.exists(dataset_path):
                    print(f"Error: Dataset file not found at {dataset_path}")  # Debug print
                    return
                    
                self.dataset = pd.read_csv(dataset_path, encoding='utf-8')
                print(f"Successfully loaded dataset with {len(self.dataset)} questions")
                print(f"Dataset columns: {self.dataset.columns.tolist()}")  # Debug print
            except Exception as e:
                print(f"Failed to load dataset: {str(e)}")
                print(f"Error type: {type(e).__name__}")  # Debug print

    @ell.complex(model="gpt-4-turbo-preview", tools=[check_readiness], temperature=0.7)
    def assess_readiness(self, message_history: List[ell.Message]) -> List[ell.Message]:
        return [
            ell.system("""You are an encouraging AI tutor that can respond in both English and Korean.
            Analyze user's response to determine if they're ready to study.
            Use the check_readiness tool to evaluate their readiness."""),
        ] + message_history

    @ell.complex(model="gpt-4-turbo-preview", tools=[generate_question], temperature=0.7)
    def question_generator(self, message_history: List[ell.Message]) -> List[ell.Message]:
        return [
            ell.system(f"""Generate an appropriate {self.difficulty} level question about {self.subject}.
            Use the generate_question tool to create a question.
            If dataset question is available, use that instead of generating new one."""),
        ] + message_history

    @ell.complex(model="gpt-4-turbo-preview", tools=[evaluate_answer], temperature=0.7)
    def answer_evaluator(self, message_history: List[ell.Message]) -> List[ell.Message]:
        return [
            ell.system("""Evaluate the user's answer considering both English and Korean:
            1. Check for correctness (considering alternative phrasings)
            2. Provide constructive feedback
            3. Give appropriate hint if wrong
            Return as JSON object."""),
            ell.user(message_history)
        ]

    def get_dataset_question(self) -> Optional[Dict]:
        """Retrieves a question from the loaded dataset."""
        if self.dataset is not None and not self.dataset.empty:
            return self.dataset.sample(n=1).iloc[0].to_dict()
        return None

    def load_questions(self):
        """Generate questions using LLM or load from dataset"""
        for _ in range(self.max_questions):
            dataset_question = self.get_dataset_question()
            print(f"Debug - Dataset question: {dataset_question}")  # Add debug logging
            
            role, content_list = self.question_generator([
                ell.user(json.dumps({
                    "command": "generate",
                    "dataset_question": dataset_question,
                    "subject": self.subject,
                    "difficulty": self.difficulty
                }))
            ])
            
            try:
                if content_list and len(content_list) > 0:
                    content_block = content_list[0]
                    if hasattr(content_block, 'tool_call'):
                        result = content_block.tool_call.tool(
                            subject=self.subject,
                            difficulty=self.difficulty,
                            dataset_question=dataset_question
                        )
                        print(f"Debug - Question generation result: {result}")  # Add debug logging
                        question_data = json.loads(result)
                        metadata = dataset_question if dataset_question else {}
                        self.questions.append(Question(
                            question_data["question"],
                            question_data["answer"],
                            question_data["hints"],
                            metadata
                        ))
                        print(f"Debug - Question added successfully")  # Add debug logging
            except Exception as e:
                print(f"Debug - Question generation error: {str(e)}")
                continue  # Continue to next question if one fails

    def converse(self, user_input: str) -> str:
        """Main conversation function that handles the interaction"""
        messages = [ell.user(user_input)]
        role_tuple, content_tuple = self.assess_readiness(messages)
        
        _, content_list = content_tuple
        
        if content_list and len(content_list) > 0:
            content_block = content_list[0]
            if hasattr(content_block, 'tool_call'):
                result = content_block.tool_call.tool(user_input)
                return result
            
        return "I couldn't understand that response."

    def check_user_readiness(self, user_input: str) -> str:
        try:
            response = self.converse(user_input)
            outer_json = json.loads(response)
            inner_json_str = outer_json["message"].strip('`json\n')
            inner_json = json.loads(inner_json_str)
            
            if inner_json["is_ready"]:
                if not self.questions:
                    try:
                        self.load_questions()
                        if not self.questions:
                            return "I'm having trouble generating questions. Please try again."
                    except Exception as e:
                        print(f"Debug - Question loading error: {str(e)}")
                        return "I'm having trouble generating questions. Please try again."
                
                self.state = UserState.READY
                self.current_question = self.questions[self.current_question_index]
                return f"{inner_json['message']}\n{self.get_current_question()}"
            
            return inner_json["message"]
            
        except Exception as e:
            print(f"Debug - Error: {str(e)}")
            return "I couldn't understand that response. Are you ready to start learning?"

    def get_current_question(self) -> str:
        if self.current_question_index >= len(self.questions):
            self.state = UserState.FINISHED
            return "Congratulations! You've completed all questions!"
        
        self.current_question = self.questions[self.current_question_index]
        metadata = self.current_question.metadata
        
        # Include context from dataset if available
        context = ""
        if metadata:
            context = f"[{metadata.get('학년', '')} {metadata.get('과목', '')} - {metadata.get('대단원', '')}]\n"
        
        return f"{context}{self.current_question.question}"

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

        if self.state in [UserState.READY, UserState.ANSWERING] and self.current_question:
            return self.check_answer(user_input)
        else:
            self.state = UserState.NOT_READY
            return "Let's start over. Are you ready to begin?"

def main():
    dataset_path = r"C:\Users\DNSOFT\Downloads\AI튜터 프롬프트 테스트 - 대화셋.csv"
    tutor = AITutor(dataset_path)
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
    