import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import HfFolder

# Hugging Face token login function
def login_to_hf(token):
    HfFolder.save_token(token)

# Request body schema
class ImageQueryRequest(BaseModel):
    image_base64: str
    user_query: str

# Initialize the FastAPI app
app = FastAPI()

# Load the model and processor when the app starts
@app.on_event("startup")
async def load_model():
    global model, processor
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    try:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(f"Error loading model or processor: {str(e)}")

# Define the prediction endpoint
@app.post("/predict")
async def predict_image(request: ImageQueryRequest):
    try:
        # Decode the base64-encoded image
        try:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data))
        except (base64.binascii.Error, UnidentifiedImageError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        # Prepare input for the model
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": request.user_query}
            ]}
        ]
        
        # Process the input and generate the model output
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, input_text, return_tensors="pt").to(model.device)

        # Generate output from the model
        output = model.generate(**inputs, max_new_tokens=500)

        # Decode the output and return the result
        decoded_output = processor.decode(output[0], skip_special_tokens=True)
        return {"output": decoded_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Example for Hugging Face token login
HF_TOKEN = "<>"
login_to_hf(HF_TOKEN)
