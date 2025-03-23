
# Image Embeddings - Textbook Discoverability

## Overview

This repository provides instructions and scripts to host Image retrieval COLPALI model script (includes endpoints for different embedding and retrieval functionalities) and two large language models with vision capabilities: **LLaMA 3.2 Vision** and **Paligemma**. These models can be hosted on a server to provide inference capabilities through an API.

## Prerequisites

Before proceeding, ensure the following are installed on your server:

- Python 3.8+
- Pip (Python package manager)

## Steps to Host the Models

Follow these steps to get the models running on your server.

### 1. Clone the Repository

First, clone the repository to your server using the following command:

```
git clone <repository-url>
```

### 2. Install Required Packages

Install all necessary dependencies by executing the command below:

```
pip install -r requirements.txt
```

### 3. Navigate to the Model Directory

In order to host colpali model and functionalities script, move into the below mentioned directory:

```
cd colpali
```

In order to host LLM Vision Model for inferencing(User response generation) move into the below mentioned directory:

```
cd llm_vision_models
```

### 4. Run the Model Hosting Scripts

To start the API for each model, use the following commands:

#### i. Colpali Image Retrieval Model and functionalities

Start the server for **Colpali Image Retrieval Model** using this command:

```
python -m uvicorn colpali_host_script.py:app --host 0.0.0.0 --port 8000 --reload
```

#### ii. LLaMA 3.2 Vision

Start the server for **LLaMA 3.2 Vision** using this command:

```
python -m uvicorn llama_3_2_vision_host_script.py:app --host 0.0.0.0 --port 8000 --reload
```

#### iii. Paligemma

Start the server for **Paligemma** using this command:

```
python -m uvicorn paligemma_host_script.py:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Accessing the Models

Once the server is running, the models will be accessible via the provided API endpoints:
- **Colpali**: `http://<server-ip>:8000`
- **LLaMA 3.2 Vision**: `http://<server-ip>:8000`
- **Paligemma**: `http://<server-ip>:8000`

Use the relevant endpoint for inference tasks and interacting with the models.
