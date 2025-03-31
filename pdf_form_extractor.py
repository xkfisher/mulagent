import json
import boto3
from typing import Dict, Any, Optional
import PyPDF2
from pathlib import Path
import argparse
from datetime import datetime
from botocore.exceptions import ClientError

class BedrockAgentManager:
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize the Bedrock Agent manager"""
        self.bedrock_agent = boto3.client(
            service_name='bedrock-agent',
            region_name=region_name
        )
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        self.agent_id = None
        self.agent_name = "PDFFormExtractorAgent"

    def create_or_get_agent(self) -> str:
        """Create a new agent or get existing one"""
        try:
            # List existing agents
            response = self.bedrock_agent.list_agents()
            
            # Check if our agent exists
            for agent in response.get('agentSummaries', []):
                if agent['agentName'] == self.agent_name:
                    self.agent_id = agent['agentId']
                    return self.agent_id
            
            # Create new agent if not found
            create_response = self.bedrock_agent.create_agent(
                agentName=self.agent_name,
                foundationModel='anthropic.claude-3-sonnet-20240229-v1:0',
                description='Agent for extracting structured data from PDF forms',
                roleArn='arn:aws:iam::YOUR_ACCOUNT_ID:role/BedrockAgentRole',  # Replace with your role ARN
                agentResourceRoleArn='arn:aws:iam::YOUR_ACCOUNT_ID:role/BedrockAgentResourceRole'  # Replace with your resource role ARN
            )
            
            self.agent_id = create_response['agent']['agentId']
            return self.agent_id
            
        except ClientError as e:
            raise Exception(f"Error managing Bedrock agent: {str(e)}")

    def create_action_group(self, agent_id: str):
        """Create action group for form processing"""
        try:
            self.bedrock_agent.create_agent_action_group(
                agentId=agent_id,
                agentVersion='DRAFT',
                actionGroupName='FormProcessing',
                description='Actions for processing PDF forms',
                apiSchema={
                    'openapi': '3.0.0',
                    'info': {
                        'title': 'Form Processing API',
                        'version': '1.0.0'
                    },
                    'paths': {
                        '/process-form': {
                            'post': {
                                'summary': 'Process PDF form',
                                'operationId': 'processForm',
                                'requestBody': {
                                    'content': {
                                        'application/json': {
                                            'schema': {
                                                'type': 'object',
                                                'properties': {
                                                    'pdf_text': {
                                                        'type': 'string',
                                                        'description': 'Text extracted from PDF form'
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            )
        except ClientError as e:
            raise Exception(f"Error creating action group: {str(e)}")

    def prepare_agent(self, agent_id: str):
        """Prepare the agent for use"""
        try:
            self.bedrock_agent.prepare_agent(
                agentId=agent_id,
                agentVersion='DRAFT'
            )
        except ClientError as e:
            raise Exception(f"Error preparing agent: {str(e)}")

    def invoke_agent(self, pdf_text: str) -> Dict[str, Any]:
        """Invoke the agent to process form text"""
        try:
            if not self.agent_id:
                self.agent_id = self.create_or_get_agent()
                self.create_action_group(self.agent_id)
                self.prepare_agent(self.agent_id)

            response = self.bedrock_runtime.invoke_agent(
                agentId=self.agent_id,
                agentAliasId='TSTALIASID',  # Replace with your alias ID
                sessionId=str(datetime.now().timestamp()),
                inputText=f"""
                Please analyze the following form text and extract all fields and their values into a structured JSON format.
                
                Guidelines:
                1. Identify all form fields and their values
                2. Maintain the hierarchical structure of the form
                3. Use null for empty or missing values
                4. Preserve field names as they appear in the form
                5. Group related fields under appropriate categories
                6. Ensure the output is valid JSON without any additional text or markdown formatting
                
                Form Text:
                {pdf_text}
                """
            )
            
            return json.loads(response['completion'])
            
        except ClientError as e:
            raise Exception(f"Error invoking agent: {str(e)}")

class PDFFormProcessor:
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize the form processor with Bedrock agent manager"""
        self.agent_manager = BedrockAgentManager(region_name)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def process_form(self, pdf_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a form and optionally save the output"""
        try:
            # Extract text from PDF
            pdf_text = self.extract_text_from_pdf(pdf_path)
            
            # Process with Bedrock agent
            result = self.agent_manager.invoke_agent(pdf_text)
            
            # Add metadata
            result['metadata'] = {
                'processed_at': datetime.now().isoformat(),
                'pdf_path': pdf_path,
                'agent_id': self.agent_manager.agent_id,
                'version': '1.0.0'
            }
            
            # Save to file if output path is provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to {output_path}")
            
            return result
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Extract structured data from PDF forms using Amazon Bedrock Agent')
    parser.add_argument('pdf_path', help='Path to the PDF form file')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    parser.add_argument('--region', '-r', default='us-east-1', help='AWS region (default: us-east-1)')
    
    args = parser.parse_args()
    
    try:
        # Initialize the processor
        processor = PDFFormProcessor(region_name=args.region)
        
        # Process the form
        result = processor.process_form(args.pdf_path, args.output)
        
        # Print results if not saving to file
        if not args.output:
            print("Extracted Form Data:")
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 