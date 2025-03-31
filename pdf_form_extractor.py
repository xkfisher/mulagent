import json
import boto3
import base64
from typing import Dict, Any
import PyPDF2
from pathlib import Path
import argparse

class PDFFormExtractor:
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize the PDF Form Extractor with Bedrock client"""
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def encode_pdf_to_base64(self, pdf_path: str) -> str:
        """Encode PDF file to base64 string"""
        with open(pdf_path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')

    def process_form_with_bedrock(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF form using Amazon Bedrock and return structured data"""
        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf(pdf_path)
        
        # Prepare the prompt for Bedrock
        prompt = f"""
        Please analyze the following form text and extract all fields and their values into a structured JSON format.
        If a field is empty or not present, use null as the value.
        Ensure the output is valid JSON without any additional text or markdown formatting.
        
        Form Text:
        {pdf_text}
        """
        
        # Prepare the request body for Bedrock
        request_body = {
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.1,
            "top_p": 0.95,
            "stop_sequences": ["\n\n"],
        }
        
        # Call Bedrock API
        response = self.bedrock.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',  # Using Claude Sonnet 3.5
            body=json.dumps(request_body)
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        # Extract the generated JSON from the response
        try:
            # The response might contain markdown code blocks, so we need to clean it
            json_str = response_body['completion'].strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {"error": "Failed to parse JSON response"}

def main():
    parser = argparse.ArgumentParser(description='Extract structured data from PDF forms using Amazon Bedrock')
    parser.add_argument('pdf_path', help='Path to the PDF form file')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    parser.add_argument('--region', '-r', default='us-east-1', help='AWS region (default: us-east-1)')
    
    args = parser.parse_args()
    
    try:
        # Initialize the extractor
        extractor = PDFFormExtractor(region_name=args.region)
        
        # Process the form
        result = extractor.process_form_with_bedrock(args.pdf_path)
        
        # Handle output
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print("Extracted Form Data:")
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error processing form: {e}")

if __name__ == "__main__":
    main() 