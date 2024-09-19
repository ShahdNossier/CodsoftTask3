from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load pre-trained model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image from a local file
image_path = r"C:\Users\Admin\Downloads\random cat.jpeg"  # Replace with your local image path
image = Image.open(image_path)

# Preprocess and generate caption
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print(caption)
