# MLOps Generative AI on AWS - Comprehensive Guide

A production-ready MLOps project demonstrating deployment and inference of Large Language Models (LLMs) using AWS services including Amazon SageMaker, Amazon Bedrock, AWS Lambda, and S3. This repository showcases best practices for deploying both lightweight and large-scale AI models with prompt engineering techniques.

## 📋 Overview

This project provides end-to-end solutions for deploying AI/ML models on AWS infrastructure with three main implementations:

1. **Lightweight Question-Answering System** - Deploy DistilBERT for context-based Q&A on SageMaker
2. **Large Language Model Deployment** - Host Falcon-40B-Instruct with advanced prompt engineering
3. **Serverless Content Generation** - Automated blog generation using Bedrock and Lambda

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    AWS Cloud Infrastructure                   │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────────────┐         ┌─────────────────────┐      │
│  │  Amazon SageMaker  │         │   AWS Lambda +      │      │
│  │  ┌──────────────┐  │         │   Amazon Bedrock    │      │
│  │  │ DistilBERT   │  │         │  ┌───────────────┐  │      │
│  │  │ Q&A Model    │  │         │  │ Blog Generator│  │      │
│  │  │ ml.m5.xlarge │  │         │  │ Llama3-2-3B   │  │      │
│  │  └──────────────┘  │         │  └───────────────┘  │      │
│  │                    │         │         │            │      │
│  │  ┌──────────────┐  │         │         ▼            │      │
│  │  │ Falcon-40B   │  │         │  ┌───────────────┐  │      │
│  │  │ Instruct     │  │         │  │  API Gateway  │  │      │
│  │  │ml.g5.12xlarge│  │         │  └───────────────┘  │      │
│  │  │  4x A10G GPU │  │         └─────────────────────┘      │
│  │  └──────────────┘  │                    │                  │
│  └────────────────────┘                    │                  │
│           │                                 │                  │
│           ▼                                 ▼                  │
│  ┌──────────────────────────────────────────────────┐         │
│  │            Amazon S3 Storage                     │         │
│  │  - Models & Artifacts                            │         │
│  │  - Generated Content (blog-output/)              │         │
│  └──────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
mlopsgenerativeaiaws/
├── aws_sagemaker/
│   ├── jupyterlab_aws.ipynb        # DistilBERT deployment for Q&A
│   └── jupyterlab_aws_full.ipynb   # Falcon-40B-Instruct with prompt engineering
├── app.py                          # Lambda function for Bedrock blog generation
├── requirements.txt                # Python dependencies (boto3)
├── .gitignore                      # Git ignore rules
└── README.md                       # Comprehensive documentation
```

## 🚀 Components

### 1. SageMaker - Question Answering System
**File:** [aws_sagemaker/jupyterlab_aws.ipynb](aws_sagemaker/jupyterlab_aws.ipynb)

Lightweight NLP model for extractive question-answering tasks using DistilBERT.

**Model Details:**
- **HuggingFace Model ID:** `distilbert-base-uncased-distilled-squad`
- **Task Type:** Question-Answering (Extractive)
- **Instance Type:** `ml.m5.xlarge`
- **Framework:** Transformers 4.26 + PyTorch 1.13 + Python 3.9

**Key Features:**
- Context-based question answering
- Fast inference (~100-200ms per request)
- Cost-effective deployment
- Production-ready endpoint with auto-scaling

**Implementation Highlights:**
```python
from sagemaker.huggingface.model import HuggingFaceModel

hub = {
    'HF_MODEL_ID': 'distilbert-base-uncased-distilled-squad',
    'HF_TASK': 'question-answering'
}

huggingface_model = HuggingFaceModel(
    env=hub,
    role=role,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version='py39'
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)
```

**Usage Example:**
```python
data = {
    "inputs": {
        "question": "What does Yaro do?",
        "context": "My Name is Yaro and I am a developer."
    }
}
response = predictor.predict(data)
# Returns: {"answer": "developer", "score": 0.95, ...}
```

---

### 2. SageMaker - Large Language Model (Falcon-40B)
**File:** [aws_sagemaker/jupyterlab_aws_full.ipynb](aws_sagemaker/jupyterlab_aws_full.ipynb)

Advanced deployment of Falcon-40B-Instruct for text generation with comprehensive prompt engineering examples.

**Model Details:**
- **HuggingFace Model ID:** `tiiuae/falcon-40b-instruct`
- **Model Size:** 40 billion parameters
- **Instance Type:** `ml.g5.12xlarge`
- **GPU Configuration:** 4x NVIDIA A10G GPUs (96GB total GPU memory)
- **Inference Engine:** Text Generation Inference (TGI)
- **License:** Apache 2.0

**Configuration Parameters:**
```python
config = {
    'HF_MODEL_ID': "tiiuae/falcon-40b-instruct",
    'SM_NUM_GPUS': 4,
    'MAX_INPUT_LENGTH': 1024,      # Maximum input tokens
    'MAX_TOTAL_TOKENS': 2048,      # Max input + output tokens
    # 'HF_MODEL_QUANTIZE': "bitsandbytes"  # Optional: Enable quantization
}
```

**Supported Generation Parameters:**
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `temperature` | Controls randomness/creativity | 1.0 | 0.0-2.0 |
| `top_p` | Nucleus sampling threshold | null | 0.0-1.0 |
| `top_k` | Top-K filtering | null | 1-100 |
| `max_new_tokens` | Maximum tokens to generate | 20 | 1-512 |
| `repetition_penalty` | Penalize repetitions | null | 1.0-2.0 |
| `do_sample` | Use sampling vs greedy | false | bool |
| `stop` | Stop sequences | [] | array |

**Use Cases Demonstrated:**
1. **Conversational AI** - Multi-turn dialogue systems
2. **Contextual Q&A** - Answer questions based on provided context
3. **Few-Shot Learning** - Sentiment analysis with examples
4. **Structured Prompting** - Controlled output generation

---

### 3. AWS Lambda + Bedrock - Serverless Blog Generator
**File:** [app.py](app.py)

Fully serverless blog content generation using Amazon Bedrock with automatic S3 storage.

**Architecture Components:**
- **Compute:** AWS Lambda (serverless)
- **AI Service:** Amazon Bedrock
- **Model:** `meta.llama3-2-3b-instruct-v1:0`
- **Storage:** Amazon S3 (`awsbedrock3` bucket)
- **API:** API Gateway (REST API)

**Key Features:**
- Zero server management
- Pay-per-invocation pricing
- Automatic scaling
- Built-in retry logic (max 3 attempts)
- 300-second timeout for long generations
- Timestamped output files

**Function Workflow:**
```
API Request → Lambda Handler → Bedrock Invoke → Generate Blog → Save to S3 → Return Response
```

**Implementation Details:**
```python
def blog_generate_using_bedrock(blogtopic: str) -> str:
    prompt = f"""<s>[INST]Human: Write a 200 words blog on the topic {blogtopic}
    Assistant:[/INST]
    """

    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.6,
        "top_p": 0.9
    }

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name="eu-west-3",
        config=botocore.config.Config(
            read_timeout=300,
            retries={'max_attempts': 3}
        )
    )

    response = bedrock.invoke_model(
        body=json.dumps(body),
        modelId="meta.llama3-2-3b-instruct-v1:0"
    )
```

**S3 Storage Pattern:**
- **Bucket:** `awsbedrock3`
- **Key Format:** `blog-output/{HHMMSS}.txt`
- **Content:** Plain text blog content

**API Request Format:**
```bash
curl -X POST https://{api-id}.execute-api.{region}.amazonaws.com/prod/blog \
  -H "Content-Type: application/json" \
  -d '{"blog_topic": "Machine Learning on AWS"}'
```

**Response:**
```json
{
    "statusCode": 200,
    "body": "\"Blog Generation is completed\""
}
```

## 🛠️ Installation and Setup

### Prerequisites

**AWS Account Requirements:**
- Active AWS account with billing enabled
- AWS CLI configured with credentials
- Appropriate service quotas for GPU instances (for Falcon-40B)

**Required IAM Permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:*",
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "iam:GetRole",
        "iam:PassRole",
        "lambda:CreateFunction",
        "lambda:InvokeFunction",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

**Local Development:**
- Python 3.9 or higher
- Git for version control
- SageMaker Studio, JupyterLab, or local Jupyter environment

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/mlopsgenerativeaiaws.git
cd mlopsgenerativeaiaws
```

---

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt

# For SageMaker notebooks, also install:
pip install sagemaker --upgrade
pip install boto3 --upgrade
```

**Dependencies:**
- `boto3` - AWS SDK for Python
- `sagemaker` - SageMaker Python SDK (for notebooks)
- `botocore` - AWS service client library

---

### Step 3: Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter default region (e.g., us-east-1, eu-west-3)
# Enter default output format (json)
```

---

### Step 4: Deploy SageMaker Models

#### Option A: Using SageMaker Studio (Recommended)

1. **Open SageMaker Studio Console**
   ```bash
   aws sagemaker create-presigned-domain-url \
     --domain-id <your-domain-id> \
     --user-profile-name <your-profile-name>
   ```

2. **Upload Notebooks**
   - Upload `aws_sagemaker/jupyterlab_aws.ipynb`
   - Upload `aws_sagemaker/jupyterlab_aws_full.ipynb`

3. **Execute Notebooks**
   - Open the desired notebook
   - Select kernel: Python 3 (Data Science)
   - Run all cells sequentially
   - Save the endpoint name for future use

#### Option B: Using Local Jupyter

```bash
jupyter notebook aws_sagemaker/
```

**Important Notes:**
- DistilBERT deployment takes ~5-10 minutes
- Falcon-40B deployment takes ~15-20 minutes (large model download)
- Ensure you have sufficient service quotas for `ml.g5.12xlarge` instances

---

### Step 5: Deploy Lambda Function for Blog Generation

#### Create S3 Bucket

```bash
aws s3 mb s3://awsbedrock3 --region eu-west-3
```

#### Create IAM Role for Lambda

```bash
# Create trust policy file
cat > lambda-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
  --role-name lambda-bedrock-blog-role \
  --assume-role-policy-document file://lambda-trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name lambda-bedrock-blog-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-bedrock-blog-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess

aws iam attach-role-policy \
  --role-name lambda-bedrock-blog-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

#### Package and Deploy Lambda Function

```bash
# Create deployment package
zip blog-generator.zip app.py

# Deploy the function
aws lambda create-function \
  --function-name blog-generator \
  --runtime python3.9 \
  --role arn:aws:iam::YOUR-ACCOUNT-ID:role/lambda-bedrock-blog-role \
  --handler app.lambda_handler \
  --zip-file fileb://blog-generator.zip \
  --timeout 300 \
  --memory-size 512 \
  --region eu-west-3
```

#### Create API Gateway (Optional)

```bash
# Create REST API
aws apigateway create-rest-api \
  --name blog-generation-api \
  --region eu-west-3

# Configure API Gateway to trigger Lambda
# (Follow AWS documentation for detailed API Gateway setup)
```

---

### Step 6: Enable Amazon Bedrock Model Access

1. Navigate to Amazon Bedrock console
2. Go to "Model access" in the left sidebar
3. Click "Manage model access"
4. Enable access to `Llama 3.2 3B Instruct`
5. Submit access request (usually approved instantly)

## 📝 Usage Examples

### 1. Question-Answering with DistilBERT

**Basic Q&A:**
```python
# Using the deployed SageMaker endpoint
data = {
    "inputs": {
        "question": "What does Yaro do?",
        "context": "My Name is Yaro and I am a developer."
    }
}

response = predictor.predict(data)
print(response)
# Output: {'score': 0.95, 'start': 30, 'end': 39, 'answer': 'developer'}
```

**Domain-Specific Q&A:**
```python
data = {
    "inputs": {
        "question": "What is used for inference?",
        "context": "This model is deployed on Amazon SageMaker for inference tasks."
    }
}

response = predictor.predict(data)
print(f"Answer: {response['answer']}")
print(f"Confidence: {response['score']:.2%}")
```

---

### 2. Text Generation with Falcon-40B

#### Simple Conversational AI

```python
prompt = """You are an helpful Assistant, called Falcon. Knowing everything about AWS.

User: Can you tell me something about Amazon SageMaker?
Falcon:"""

payload = {
    "inputs": prompt,
    "parameters": {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_new_tokens": 1024,
        "repetition_penalty": 1.03,
        "stop": ["\nUser:", "<|endoftext|>", "</s>"]
    }
}

response = llm.predict(payload)
for seq in response:
    print(f"Result: {seq['generated_text']}")
```

#### Structured Prompt Engineering

**Example: Context-Based Question Answering**

```python
prompt = """
Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.

Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.

Question: What was OKT3 originally sourced from?

Answer:"""

payload = {
    "inputs": prompt,
    "parameters": {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_new_tokens": 1024,
        "repetition_penalty": 1.03,
        "stop": ["\nUser:", "<|endoftext|>", "</s>"]
    }
}

response = llm.predict(payload)
for seq in response:
    print(f"Answer: {seq['generated_text']}")
# Output: "Mice"
```

#### Few-Shot Learning for Sentiment Analysis

```python
prompt = """
Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredible"
Sentiment:"""

payload = {
    "inputs": prompt,
    "parameters": {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_new_tokens": 50,
        "repetition_penalty": 1.03,
        "stop": ["###", "\n"]
    }
}

response = llm.predict(payload)
for seq in response:
    print(f"Result: {seq['generated_text']}")
# Output: "Positive"
```

---

### 3. Blog Generation via Lambda + Bedrock

#### Direct Lambda Invocation

```bash
aws lambda invoke \
  --function-name blog-generator \
  --payload '{"body": "{\"blog_topic\": \"Machine Learning on AWS\"}"}' \
  --region eu-west-3 \
  response.json

cat response.json
```

#### API Gateway REST API Call

```bash
curl -X POST https://{api-id}.execute-api.eu-west-3.amazonaws.com/prod/blog \
  -H "Content-Type: application/json" \
  -d '{"blog_topic": "Artificial Intelligence and AWS"}'
```

#### Python Client Example

```python
import requests
import json

url = "https://{api-id}.execute-api.eu-west-3.amazonaws.com/prod/blog"
headers = {"Content-Type": "application/json"}
data = {"blog_topic": "The Future of Generative AI"}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

#### Retrieve Generated Blog from S3

```bash
# List generated blogs
aws s3 ls s3://awsbedrock3/blog-output/

# Download a specific blog
aws s3 cp s3://awsbedrock3/blog-output/{timestamp}.txt ./
```

---

### 4. Testing Lambda Function Locally

```python
import json
from app import lambda_handler

# Simulate API Gateway event
event = {
    "body": json.dumps({"blog_topic": "Cloud Computing with AWS"})
}

context = {}  # Mock context object

# Invoke handler
result = lambda_handler(event, context)
print(json.dumps(result, indent=2))
```

## 💰 Cost Estimation & Optimization

### Pricing Breakdown (as of 2025)

| Service | Resource Type | Pricing Model | Approximate Cost |
|---------|--------------|---------------|------------------|
| **SageMaker - DistilBERT** | ml.m5.xlarge | Per-hour | $0.23/hour |
| **SageMaker - Falcon-40B** | ml.g5.12xlarge | Per-hour | $7.09/hour |
| **Amazon Bedrock** | Llama 3.2 3B | Per-token | $0.0008/1K input tokens<br>$0.0024/1K output tokens |
| **AWS Lambda** | 512MB memory | Per-invocation + compute time | $0.20/1M requests<br>+ $0.0000166667/GB-sec |
| **Amazon S3** | Standard storage | Per GB/month | $0.023/GB |
| **API Gateway** | REST API | Per million requests | $3.50/million |

### Monthly Cost Examples

**Scenario 1: Development/Testing**
- DistilBERT endpoint: 8 hours/day = ~$55/month
- Lambda: 1000 invocations/day = ~$0.60/month
- S3 storage: 1GB = ~$0.02/month
- **Total: ~$56/month**

**Scenario 2: Production with Falcon-40B**
- Falcon-40B endpoint: 24/7 uptime = ~$5,185/month
- Lambda: 10,000 invocations/day = ~$6/month
- S3 storage: 100GB = ~$2.30/month
- **Total: ~$5,193/month**

### Cost Optimization Strategies

1. **Use Auto-Scaling for SageMaker Endpoints**
   ```python
   predictor.update_endpoint(
       initial_instance_count=1,
       instance_type='ml.m5.xlarge',
       wait=True
   )

   # Configure auto-scaling
   client = boto3.client('application-autoscaling')
   client.register_scalable_target(
       ServiceNamespace='sagemaker',
       ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
       ScalableDimension='sagemaker:variant:DesiredInstanceCount',
       MinCapacity=1,
       MaxCapacity=3
   )
   ```

2. **Delete Endpoints When Not in Use**
   ```python
   # Always clean up after testing
   predictor.delete_model()
   predictor.delete_endpoint()
   ```

3. **Use Serverless Inference for Low Traffic**
   - Consider SageMaker Serverless Inference for <200 requests/hour
   - Pay only for compute time used

4. **Enable S3 Lifecycle Policies**
   ```bash
   # Move old blog outputs to Glacier after 30 days
   aws s3api put-bucket-lifecycle-configuration \
     --bucket awsbedrock3 \
     --lifecycle-configuration file://lifecycle.json
   ```

5. **Monitor with AWS Budgets**
   ```bash
   aws budgets create-budget \
     --account-id 123456789012 \
     --budget file://budget.json \
     --notifications-with-subscribers file://notifications.json
   ```

---

## 🔧 Prompt Engineering Techniques

This project demonstrates advanced prompt engineering patterns for optimal LLM performance.

### 1. Structured Prompt Template

**Pattern:**
```
[Instruction] + [Context] + [Input] + [Output Indicator]
```

**Example:**
```python
prompt = """
Answer the question based on the context below. Keep the answer short and concise.

Context: {context_information}

Question: {user_question}

Answer:"""
```

**Benefits:**
- Clear task definition
- Reduces hallucination
- Improves answer quality
- Easier to debug

---

### 2. Few-Shot Learning

Provide examples to guide model behavior without fine-tuning.

**Pattern:**
```
Example 1: Input → Output
Example 2: Input → Output
Example 3: Input → Output
---
New Input: ? →
```

**Implementation:**
```python
prompt = """
Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "{new_tweet}"
Sentiment:"""
```

**Use Cases:**
- Classification tasks
- Format specification
- Style transfer
- Custom output structures

---

### 3. Generation Control Parameters

Fine-tune model behavior with these parameters:

| Parameter | Purpose | Recommended Range | Effect |
|-----------|---------|-------------------|--------|
| `temperature` | Creativity vs consistency | 0.7-0.9 (creative)<br>0.1-0.3 (factual) | Higher = more random |
| `top_p` | Nucleus sampling | 0.9-0.95 | Limits token pool to cumulative probability |
| `top_k` | Vocabulary filtering | 40-50 | Limits to top K tokens |
| `repetition_penalty` | Avoid repetition | 1.0-1.2 | Higher = less repetition |
| `max_new_tokens` | Length control | Task-dependent | Hard limit on output length |

**Example Configuration:**
```python
# For factual Q&A
parameters = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_new_tokens": 100,
    "repetition_penalty": 1.1
}

# For creative writing
parameters = {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "repetition_penalty": 1.05
}
```

---

### 4. Stop Sequences

Control when generation should terminate:

```python
payload = {
    "inputs": prompt,
    "parameters": {
        "stop": ["\nUser:", "<|endoftext|>", "</s>", "###"]
    }
}
```

**Common Stop Patterns:**
- `\nUser:` - For chatbot turn-taking
- `###` - For structured output separation
- `</s>`, `<|endoftext|>` - Model-specific end tokens

---

### 5. Prompt Engineering Best Practices

**DO:**
- ✅ Be specific and explicit in instructions
- ✅ Provide context before asking questions
- ✅ Use examples for complex tasks
- ✅ Test different temperature settings
- ✅ Include output format specifications

**DON'T:**
- ❌ Use ambiguous language
- ❌ Mix multiple tasks in one prompt
- ❌ Assume model has recent knowledge
- ❌ Ignore token limits
- ❌ Forget to handle edge cases

## 🛡️ Security Best Practices

### 1. IAM Roles and Policies

**Principle of Least Privilege:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": "arn:aws:sagemaker:region:account:endpoint/specific-endpoint-name"
    }
  ]
}
```

### 2. Data Protection

- **Encryption at Rest:** Enable S3 bucket encryption
  ```bash
  aws s3api put-bucket-encryption \
    --bucket awsbedrock3 \
    --server-side-encryption-configuration \
    '{"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}'
  ```

- **Encryption in Transit:** All AWS services use HTTPS by default

- **VPC Endpoints:** Use VPC endpoints for SageMaker and Bedrock
  ```bash
  aws ec2 create-vpc-endpoint \
    --vpc-id vpc-xxxxx \
    --service-name com.amazonaws.region.sagemaker.runtime \
    --route-table-ids rtb-xxxxx
  ```

### 3. Input Validation

**Lambda Function Enhancement:**
```python
import re

def validate_blog_topic(topic):
    # Limit length
    if len(topic) > 200:
        raise ValueError("Topic too long")

    # Block sensitive patterns
    if re.search(r'(password|secret|api[_-]?key)', topic, re.I):
        raise ValueError("Invalid topic content")

    return topic
```

### 4. Secrets Management

**Never hardcode credentials:**
```python
# BAD
bucket_name = "awsbedrock3"  # Hardcoded

# GOOD - Use environment variables or AWS Secrets Manager
import os
bucket_name = os.environ.get('S3_BUCKET_NAME')

# BETTER - Use AWS Secrets Manager
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])
```

### 5. Monitoring and Alerts

- Enable CloudTrail for all API calls
- Set up CloudWatch Alarms for unusual activity
- Use AWS GuardDuty for threat detection
- Implement AWS Config for compliance monitoring

---

## 🔍 Monitoring and Observability

### CloudWatch Dashboards

**SageMaker Endpoint Metrics:**
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create custom dashboard
cloudwatch.put_dashboard(
    DashboardName='SageMaker-Endpoints',
    DashboardBody=json.dumps({
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["AWS/SageMaker", "ModelLatency", {"stat": "Average"}],
                        [".", "Invocations", {"stat": "Sum"}],
                        [".", "Invocation4XXErrors", {"stat": "Sum"}]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Endpoint Performance"
                }
            }
        ]
    })
)
```

### Key Metrics to Monitor

| Metric | Service | Alert Threshold | Action |
|--------|---------|-----------------|--------|
| ModelLatency | SageMaker | >2000ms | Scale up instances |
| Invocation4XXErrors | SageMaker | >5% error rate | Check inputs/model |
| Invocation5XXErrors | SageMaker | >1% error rate | Check endpoint health |
| Duration | Lambda | >250s | Optimize code |
| Errors | Lambda | >5% | Review logs |
| ThrottledInvocations | Bedrock | >0 | Request quota increase |

### Logging Best Practices

**Enhanced Lambda Logging:**
```python
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    # Log incoming request
    logger.info("Received event", extra={
        "event": json.dumps(event),
        "request_id": context.request_id
    })

    try:
        # Your code here
        result = blog_generate_using_bedrock(blogtopic)
        logger.info("Blog generated successfully", extra={
            "blog_length": len(result),
            "topic": blogtopic
        })
        return result
    except Exception as e:
        logger.error("Error generating blog", extra={
            "error": str(e),
            "topic": blogtopic
        }, exc_info=True)
        raise
```

### X-Ray Tracing

Enable AWS X-Ray for distributed tracing:
```python
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

patch_all()

@xray_recorder.capture('blog_generation')
def blog_generate_using_bedrock(blogtopic):
    # Your code here
    pass
```

---

## 🚨 Resource Cleanup

**CRITICAL: Always clean up resources to avoid unexpected charges!**

### Delete SageMaker Endpoints

**Option 1: From Jupyter Notebook**
```python
# Delete endpoint and model
predictor.delete_model()
predictor.delete_endpoint()

# Verify deletion
import boto3
sm_client = boto3.client('sagemaker')

try:
    sm_client.describe_endpoint(EndpointName='your-endpoint-name')
    print("Endpoint still exists!")
except:
    print("Endpoint successfully deleted")
```

**Option 2: Using AWS CLI**
```bash
# List all endpoints
aws sagemaker list-endpoints

# Delete specific endpoint
aws sagemaker delete-endpoint --endpoint-name huggingface-pytorch-inference-xxxx

# Delete endpoint configuration
aws sagemaker delete-endpoint-config --endpoint-config-name huggingface-pytorch-inference-xxxx

# Delete model
aws sagemaker delete-model --model-name huggingface-pytorch-inference-xxxx
```

### Delete Lambda Function

```bash
aws lambda delete-function --function-name blog-generator --region eu-west-3
```

### Clean Up S3 Bucket

```bash
# Delete all objects in bucket
aws s3 rm s3://awsbedrock3/blog-output/ --recursive

# Delete the bucket (if no longer needed)
aws s3 rb s3://awsbedrock3 --force
```

### Delete IAM Roles and Policies

```bash
# Detach policies
aws iam detach-role-policy \
  --role-name lambda-bedrock-blog-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Delete role
aws iam delete-role --role-name lambda-bedrock-blog-role
```

### Complete Cleanup Script

```bash
#!/bin/bash

# Set variables
ENDPOINT_NAME="your-endpoint-name"
LAMBDA_FUNCTION="blog-generator"
S3_BUCKET="awsbedrock3"
REGION="eu-west-3"

# Delete SageMaker resources
echo "Deleting SageMaker endpoint..."
aws sagemaker delete-endpoint --endpoint-name $ENDPOINT_NAME

# Delete Lambda function
echo "Deleting Lambda function..."
aws lambda delete-function --function-name $LAMBDA_FUNCTION --region $REGION

# Clean S3
echo "Cleaning S3 bucket..."
aws s3 rm s3://$S3_BUCKET/blog-output/ --recursive

echo "Cleanup complete!"
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

**1. SageMaker Endpoint Deployment Fails**

**Error:** `ResourceLimitExceeded`
```
Solution: Request service quota increase for GPU instances
aws service-quotas request-service-quota-increase \
  --service-code sagemaker \
  --quota-code L-xxxxxx \
  --desired-value 2
```

**2. Lambda Function Timeout**

**Error:** `Task timed out after 3.00 seconds`
```python
# Increase timeout in function configuration
aws lambda update-function-configuration \
  --function-name blog-generator \
  --timeout 300
```

**3. Bedrock Access Denied**

**Error:** `AccessDeniedException: User is not authorized to perform: bedrock:InvokeModel`
```
Solution: Enable model access in Bedrock console and verify IAM permissions
```

**4. S3 Permission Errors**

**Error:** `An error occurred (AccessDenied) when calling the PutObject operation`
```bash
# Verify bucket policy and IAM role permissions
aws s3api get-bucket-policy --bucket awsbedrock3
```

---

## 📚 Additional Resources

### Official AWS Documentation
- [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/)
- [AWS Lambda Developer Guide](https://docs.aws.amazon.com/lambda/)
- [Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference)

### Model Resources
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Falcon-40B-Instruct Model Card](https://huggingface.co/tiiuae/falcon-40b-instruct)
- [DistilBERT Model Documentation](https://huggingface.co/distilbert-base-uncased-distilled-squad)

### Learning Materials
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)
- [SageMaker Examples Repository](https://github.com/aws/amazon-sagemaker-examples)
- [Generative AI on AWS](https://aws.amazon.com/generative-ai/)

### Community & Support
- [AWS re:Post for SageMaker](https://repost.aws/tags/TA4ckwDWfFQZe_Ky5D6NAEIQ/amazon-sage-maker)
- [HuggingFace Forums](https://discuss.huggingface.co/)
- [AWS Developer Forums](https://forums.aws.amazon.com/)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is provided as-is for educational and demonstration purposes. Please refer to individual model licenses:
- **Falcon-40B-Instruct:** Apache 2.0 License
- **DistilBERT:** Apache 2.0 License
- **Llama 3.2:** Meta Community License

---

## 👨‍💻 Author

**Project:** MLOps Generative AI on AWS
**Repository:** [mlopsgenerativeaiaws](https://github.com/yourusername/mlopsgenerativeaiaws)
**Contact:** Open an issue for questions or support

---

## 🙏 Acknowledgments

- Technology Innovation Institute for Falcon models
- HuggingFace for model hosting and inference libraries
- AWS for comprehensive ML infrastructure
- Meta for Llama models via Amazon Bedrock

---

**⚠️ Important Reminders:**
1. Always monitor your AWS costs
2. Delete unused endpoints immediately
3. Never commit AWS credentials to version control
4. Test with small instances before scaling
5. Review AWS service quotas before deployment

