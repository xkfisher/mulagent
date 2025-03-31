import json
import boto3
from typing import Dict, Any, List, Optional
from datetime import datetime

class BedrockAgent:
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize the Bedrock agent client"""
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        self.model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'

    def invoke_model(self, prompt: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Invoke the Bedrock model with the given prompt"""
        try:
            request_body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,  # Slightly higher temperature for more natural conversation
                "top_p": 0.95,
                "stop_sequences": ["\n\n"],
            }
            
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            return json.loads(response['body'].read())
        except Exception as e:
            raise Exception(f"Error invoking Bedrock model: {str(e)}")

class FormConversationAgent:
    def __init__(self, form_data: Dict[str, Any], region_name: str = 'us-east-1'):
        """Initialize the conversation agent with form data"""
        self.form_data = form_data
        self.bedrock_agent = BedrockAgent(region_name)
        self.conversation_history: List[Dict[str, str]] = []
        
        # Initialize the conversation with form context
        self._initialize_conversation()

    def _initialize_conversation(self):
        """Initialize the conversation with form context"""
        form_summary = self._create_form_summary()
        initial_prompt = f"""
        You are a helpful assistant that can answer questions about a form. Here is the form data:
        
        {form_summary}
        
        You can help users by:
        1. Answering questions about specific fields
        2. Explaining the form structure
        3. Providing insights about the data
        4. Suggesting related information
        
        Please be concise and friendly in your responses. If you don't have enough information to answer a question, say so.
        """
        
        self.conversation_history.append({
            "role": "system",
            "content": initial_prompt
        })

    def _create_form_summary(self) -> str:
        """Create a summary of the form data"""
        summary = []
        for key, value in self.form_data.items():
            if key != 'metadata':  # Skip metadata
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        summary.append(f"{key}.{subkey}: {subvalue}")
                else:
                    summary.append(f"{key}: {value}")
        return "\n".join(summary)

    def chat(self, user_input: str) -> str:
        """Process user input and return a response"""
        try:
            # Add user input to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Create the prompt with conversation history
            prompt = self._create_conversation_prompt()
            
            # Get response from Bedrock
            response = self.bedrock_agent.invoke_model(prompt)
            
            # Extract the response
            assistant_response = response['completion'].strip()
            
            # Add assistant response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            
            return assistant_response
            
        except Exception as e:
            return f"Error processing your request: {str(e)}"

    def _create_conversation_prompt(self) -> str:
        """Create a prompt that includes the conversation history"""
        prompt_parts = []
        
        # Add system message
        prompt_parts.append(self.conversation_history[0]["content"])
        
        # Add conversation history
        for message in self.conversation_history[1:]:
            role = "Human" if message["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {message['content']}")
        
        # Add current user input
        prompt_parts.append("Human: " + self.conversation_history[-1]["content"])
        prompt_parts.append("Assistant: ")
        
        return "\n\n".join(prompt_parts)

def main():
    # Example usage
    try:
        # Load form data from JSON file
        with open('form_output.json', 'r') as f:
            form_data = json.load(f)
        
        # Initialize the conversation agent
        agent = FormConversationAgent(form_data)
        
        print("Form Conversation Agent")
        print("Type 'exit' to end the conversation")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nGoodbye!")
                break
            
            response = agent.chat(user_input)
            print(f"\nAssistant: {response}")
            
    except FileNotFoundError:
        print("Error: form_output.json not found. Please run pdf_form_extractor.py first.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 