# AWS AI/ML Suite - Model Deployment and Inference

This project demonstrates the use of AWS services to deploy and utilize artificial intelligence models, including SageMaker, Bedrock, and Lambda.

## 📋 Overview

The project consists of three main components:
1. **Model deployment on SageMaker** - Jupyter notebooks for deploying Hugging Face models
2. **Content generation with Bedrock** - Lambda function for automated blog generation
3. **S3 storage** - Automatic saving of generated content

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐
│   Jupyter Labs  │    │   AWS Lambda     │    │   S3 Bucket │
│   SageMaker     │───▶│   Bedrock        │───▶│   Storage   │
│   Endpoints     │    │   Blog Generator │    │   Output    │
└─────────────────┘    └──────────────────┘    └─────────────┘
```

## 📁 Project Structure

```
.
├── jupyterlab_aws.ipynb        # DistilBERT deployment (Q&A)
├── jupyterlab_aws_full.ipynb   # Falcon-40B deployment + Prompt Engineering
├── app.py                      # Lambda function for blog generation
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

## 🚀 Components

### 1. SageMaker - Question Answering (jupyterlab_aws.ipynb)

Deploys a **DistilBERT** model optimized for question-answering:

**Model used:** `distilbert-base-uncased-distilled-squad`
**Instance type:** `ml.m5.xlarge`
**Task:** Question-Answering

**Usage example:**
```python
data = {
    "inputs": {
        "question": "What is used for inference?",
        "context": "This model is used with sagemaker for inference."
    }
}
```

### 2. SageMaker - Large Language Model (jupyterlab_aws_full.ipynb)

Deploys **Falcon-40B-Instruct** for advanced text generation:

**Model used:** `tiiuae/falcon-40b-instruct`
**Instance type:** `ml.g5.12xlarge` (4 NVIDIA A10G GPUs)
**Capabilities:** 
- Text generation
- Prompt engineering
- Few-shot learning

**Advanced configuration:**
- MAX_INPUT_LENGTH: 1024 tokens
- MAX_TOTAL_TOKENS: 2048 tokens
- Quantization support (optional)

### 3. AWS Lambda + Bedrock (app.py)

Serverless function for automated blog generation:

**Model used:** `meta.llama3-2-3b-instruct-v1:0`
**Integrated services:**
- Amazon Bedrock (generation)
- Amazon S3 (storage)
- AWS Lambda (execution)

**Features:**
- REST API for blog generation
- Automatic S3 saving
- Robust error handling

## 🛠️ Installation and Setup

### Prerequisites

- AWS account with appropriate permissions
- Python 3.9+
- Configured AWS CLI
- SageMaker Studio or Jupyter environment

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. AWS Configuration

Ensure you have the following permissions:
- `sagemaker:*`
- `bedrock:InvokeModel`
- `s3:PutObject`
- `iam:GetRole`

### 3. SageMaker Deployment

1. Open `jupyterlab_aws.ipynb` in SageMaker Studio
2. Execute all cells sequentially
3. Note the created endpoint for future use

### 4. Lambda Deployment

```bash
# Create deployment package
zip -r blog-generator.zip app.py

# Deploy via AWS CLI
aws lambda create-function \
  --function-name blog-generator \
  --zip-file fileb://blog-generator.zip \
  --handler app.lambda_handler \
  --runtime python3.9 \
  --role arn:aws:iam::YOUR-ACCOUNT:role/lambda-execution-role
```

## 📝 Usage

### Question-Answering with DistilBERT

```python
# Using the deployed model
predictor.predict({
    "inputs": {
        "question": "Your question here",
        "context": "The context to answer the question"
    }
})
```

### Text Generation with Falcon-40B

```python
# Structured prompt for better results
prompt = """
Answer the question based on the context below. Keep the answer short and concise.

Context: [Your context]
Question: [Your question]
Answer:
"""

response = llm.predict({
    "inputs": prompt,
    "parameters": {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_new_tokens": 1024
    }
})
```

### Blog Generation via API

```bash
curl -X POST https://your-api-gateway-url/blog \
  -H "Content-Type: application/json" \
  -d '{"blog_topic": "Artificial Intelligence and AWS"}'
```

## 💰 Cost Estimation

| Service | Instance Type | Approximate Cost/Hour |
|---------|---------------|----------------------|
| SageMaker (DistilBERT) | ml.m5.xlarge | ~$0.23 |
| SageMaker (Falcon-40B) | ml.g5.12xlarge | ~$7.09 |
| Bedrock (Llama3) | Pay-per-token | ~$0.0008/1K tokens |
| Lambda | Pay-per-invocation | ~$0.20/1M requests |

## 🔧 Prompt Engineering Techniques

The project demonstrates several advanced techniques:

### 1. Structured prompts
```
Instruction + Context + Question + Output Indicator
```

### 2. Few-shot learning
```
Example 1: Input → Output
Example 2: Input → Output
Example 3: Input → Output
New case: Input → ?
```

### 3. Generation control
- `temperature`: Controls creativity
- `top_p`: Nucleus sampling
- `repetition_penalty`: Avoids repetitions

## 🛡️ Security Best Practices

- Use IAM roles with minimal permissions
- Encrypt sensitive data in transit and at rest
- Monitor costs with AWS Budgets
- Implement timeout mechanisms
- Always validate user inputs

## 🔍 Monitoring and Observability

- CloudWatch for SageMaker metrics
- X-Ray for Lambda request tracing
- CloudTrail for API call auditing
- Cost threshold alerts

## 🚨 Resource Cleanup

**Important:** Don't forget to delete SageMaker endpoints to avoid costs:

```python
# In your notebooks
predictor.delete_model()
predictor.delete_endpoint()

# Or via AWS CLI
aws sagemaker delete-endpoint --endpoint-name your-endpoint-name
```

## 📚 Additional Resources

- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Hugging Face Models](https://huggingface.co/models)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

