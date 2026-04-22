import json
from collections import Counter

# 🔹 Replace with your actual file name
file_path = "Descriptions.json"

# Load JSON
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# If JSON is wrapped inside a dictionary (sometimes happens)
if isinstance(data, dict):
    # adjust if needed, for example:
    # data = data["cases"]
    data = list(data.values())

# Extract locations
locations = []
location_categories = []

for item in data:
    loc = item.get("Location")
    loc_cat = item.get("Location Category")
    
    if loc:
        locations.append(loc)
    if loc_cat:
        location_categories.append(loc_cat)

# Convert to unique sets
unique_locations = sorted(set(locations))
unique_location_categories = sorted(set(location_categories))

# Count frequency (class distribution)
location_counts = Counter(locations)
location_category_counts = Counter(location_categories)

# 🔹 Results
print("===================================")
print("UNIQUE LOCATION")
print("===================================")
print("Number of unique Location:", len(unique_locations))
print("\nList of Locations:")
for loc in unique_locations:
    print("-", loc)

print("\n===================================")
print("UNIQUE LOCATION CATEGORY")
print("===================================")
print("Number of unique Location Category:", len(unique_location_categories))
print("\nList of Location Categories:")
for cat in unique_location_categories:
    print("-", cat)

print("\n===================================")
print("CLASS DISTRIBUTION (Location)")
print("===================================")
for loc, count in location_counts.items():
    print(f"{loc}: {count}")

print("\n===================================")
print("CLASS DISTRIBUTION (Location Category)")
print("===================================")
for cat, count in location_category_counts.items():
    print(f"{cat}: {count}")
