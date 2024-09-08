import json

# Define the input and output paths for the JSON file
input_json_path = '/jmain02/home/J2AD007/txk47/sxs27-txk47/LHGNN/datafiles/audioset_eval_wav.json'  # Replace with your actual input JSON file path
output_json_path = '/jmain02/home/J2AD007/txk47/sxs27-txk47/LHGNN/datafiles/audioset_eval_wav.json'  # Replace with your desired output JSON file path

# Define the old and new path strings
old_path = "/jmain02/flash/share/datasets/audioset/"
new_path = "/jmain02/home/J2AD007/txk47/sxs27-txk47/datasets/audioset_apocrita/"

# Load the JSON data
with open(input_json_path, 'r') as f:
    data = json.load(f)

# Update the paths
for item in data['data']:
    item['wav'] = item['wav'].replace(old_path, new_path)

# Save the updated JSON data
with open(output_json_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Paths updated and JSON saved successfully!")
