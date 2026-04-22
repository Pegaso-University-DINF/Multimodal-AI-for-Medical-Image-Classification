import os
import json
import shutil

# 🔹 Paths (CHANGE THESE)
json_path = "Descriptions.json"
source_image_folder = "images"
output_base_folder = "dataset"

# Create output base folder if not exists
os.makedirs(output_base_folder, exist_ok=True)

# Load JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Mapping to clean folder names
def clean_folder_name(name):
    return name.replace(" ", "_").replace(",", "").replace("/", "_")

# Process each entry
for item in data:
    image_id = item.get("image")
    location_category = item.get("Location Category")
    
    if not image_id or not location_category:
        continue
    
    # Clean folder name
    folder_name = clean_folder_name(location_category)
    class_folder = os.path.join(output_base_folder, folder_name)
    os.makedirs(class_folder, exist_ok=True)
    
    # Add extension
    image_filename = image_id + ".png" 
    source_path = os.path.join(source_image_folder, image_filename)
    destination_path = os.path.join(class_folder, image_filename)
    
    # Copy image if exists
    if os.path.exists(source_path):
        shutil.copy2(source_path, destination_path)
    else:
        print(f"Image not found: {source_path}")

print("Dataset organization complete.")
