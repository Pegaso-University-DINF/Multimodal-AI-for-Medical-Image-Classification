import json
import os
import pandas as pd

# Paths
description_file = "Descriptions.json"
image_base_folder = "dataset" 

# Load JSON
with open(description_file, "r") as f:
    descriptions = json.load(f)

# Collect dataset info
data = []
for item in descriptions:
    image_id = item["image"]
    caption = item["Description"]["Caption"]
    location_category = item["Location Category"]
    
    # Construct image path (assuming folder structure: images_prepared/<location_category>/<image>.jpg)
    # Adjust extension if needed
    possible_extensions = [".jpg", ".png", ".jpeg"]
    image_path = None
    for ext in possible_extensions:
        temp_path = os.path.join(image_base_folder, location_category, image_id + ext)
        if os.path.exists(temp_path):
            image_path = temp_path
            break
    
    if image_path is None:
        print(f"Image not found for text feature: {image_id}")
        continue

    data.append({
        "image_path": image_path,
        "label": location_category,
        "caption": caption
    })

# Save as CSV for later use
df = pd.DataFrame(data)
df.to_csv("multimodal_dataset.csv", index=False)
print(f"Dataset prepared with {len(df)} samples.")
