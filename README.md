# PDF Form Extractor with Amazon Bedrock

This Python tool uses Amazon Bedrock with Claude Sonnet 3.5 to extract structured data from PDF forms and convert it into JSON format.

## Features

- PDF text extraction
- Integration with Amazon Bedrock
- Structured JSON output
- Command-line interface
- Support for multiple AWS regions

## Prerequisites

- Python 3.8 or higher
- AWS account with Bedrock access
- AWS credentials configured

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-form-extractor.git
cd pdf-form-extractor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials:
Make sure you have your AWS credentials configured with appropriate permissions for Bedrock. You can do this by:
- Setting environment variables (`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`)
- Using AWS CLI configuration (`aws configure`)
- Using AWS credentials file

## Usage

Basic usage:
```bash
python pdf_form_extractor.py path/to/your/form.pdf
```

Save output to a file:
```bash
python pdf_form_extractor.py path/to/your/form.pdf -o output.json
```

Specify AWS region:
```bash
python pdf_form_extractor.py path/to/your/form.pdf -r us-west-2
```

## Output

The tool will output a JSON structure containing all form fields and their values. Empty or missing fields will be represented as `null`.

## License

MIT License 