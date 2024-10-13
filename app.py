from flask import Flask, request, jsonify
from pathlib import Path
import lancedb
from pydantic import BaseModel
import os
from typing import Optional
class SearchResult(BaseModel):
    distance: float
    name: str
    path: str
    vector: list[float]

class Result(BaseModel):
    similarity: float
    filename: str
    filetype: str
    size: int # in bytes
    thumbnail: Optional[str] # base64 encoded jpg thumbnail
    path: str # full path

class SearchResponse(BaseModel):
    results: list[Result]
    len: int




from transformers import CLIPProcessor, CLIPModel

import requests
from PIL import Image
from pathlib import Path
import torch

class ImageFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.get_image_features(pixel_values)
    
class TextFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model.get_text_features(input_ids)
    

def download_and_open_image(url, save_path):
    sample_path = Path(save_path)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url)

    with sample_path.open("wb") as f:
        f.write(r.content)

    image = Image.open(sample_path)
    return image

image_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_tulips.jpg"
save_path = "data/coco_tulips.jpg"
image = download_and_open_image(image_url, save_path)

# load pre-trained model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# load preprocessor for model input
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

app = Flask(__name__)

# Function to get text embedding (assumed to be defined elsewhere)
image_inputs = processor(
        images = image,
        return_tensors="pt"
    )

text_inputs = processor(
    text = "dog, black, in hollywood",
    return_tensors="pt"
)

import openvino as ov
from pathlib import Path
core = ov.Core()

image_fp16_model_path = Path("image_clip-vit-base-patch16.xml")
model.config.torchscript = True

if not image_fp16_model_path.exists():
    ov_model = ov.convert_model(ImageFeatureExtractor(model), example_input=dict(image_inputs))
    ov.save_model(ov_model, image_fp16_model_path)
compiled_image_model = core.compile_model(image_fp16_model_path, 'AUTO')

text_fp16_model_path = Path("text_clip-vit-base-patch16.xml")
model.config.torchscript = True

if not text_fp16_model_path.exists():
    ov_model = ov.convert_model(TextFeatureExtractor(model), example_input=dict(text_inputs))
    ov.save_model(ov_model, text_fp16_model_path)
compiled_text_model = core.compile_model(text_fp16_model_path, 'AUTO')



# Prepare the image input
image_input = processor(images=image, return_tensors="pt")["pixel_values"]

# Run inference
ov_output = compiled_image_model(image_input)
image_features = ov_output[compiled_image_model.output(0)]
print(image_features)


def get_single_image_embedding(image):
    # Get single image embeddings
    inputs = processor(
        images=image,
        return_tensors="pt"
    )
    image_input = inputs["pixel_values"]
    ov_output = compiled_image_model(image_input)
    image_features = ov_output[compiled_image_model.output(0)]
    return image_features

def get_single_text_embedding(text):
    inputs = processor(text=text, return_tensors="pt")
    text_input = inputs["input_ids"]
    ov_output = compiled_text_model(text_input)
    text_features = ov_output[compiled_text_model.output(0)]
    return text_features

# Connect to LanceDB
db = lancedb.connect("./.lancedb")

class FileInfo(BaseModel):
    filename: str
    file_type: str
    file_size: int

def get_file_info(file_path: str) -> FileInfo:
    # Create a Path object from the file path
    path = Path(file_path)
    
    # Check if the file exists
    if not path.is_file():
        raise FileNotFoundError(f"No file found at {file_path}")
    
    # Get the filename
    file_name = path.name
    
    # Get the file extension (file type)
    file_extension = path.suffix
    
    # Get the file size in bytes
    file_size = path.stat().st_size
    
    # Return a Pydantic model with the file information
    return FileInfo(
        filename=file_name,
        file_type=file_extension,
        file_size=file_size
    )

@app.route('/search', methods=['GET'])
def search():
    # Get the query parameter 'q' from the request
    query = request.args.get('q', default=None, type=str)
    
    if not query:
        return jsonify({"error": "Query parameter 'q' is required."}), 400
    
    # Get the text embedding for the query
    text_embedding = get_single_text_embedding(query)

    # Open the table in LanceDB
    tbl = db.open_table("pt_table")

    # Perform the search using the embedding
    search_results = tbl.search(query=text_embedding.tolist()[0]).limit(4).to_list()

    # change the field name from _distance to distance
    for result in search_results:
        result["distance"] = result.get("_distance")   

    search_results = [SearchResult.model_validate(result) for result in search_results]

    ret = []
    for result in search_results:
        print(result)
        file_info = get_file_info(result.path)
        ret.append(Result(similarity=result.distance/100, filename=file_info.filename, filetype=file_info.file_type, size=file_info.file_size, path=result.path, thumbnail=None))

    # Return the search results as a JSON response
    return jsonify(SearchResponse(results=ret, len=len(ret)).model_dump())

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
