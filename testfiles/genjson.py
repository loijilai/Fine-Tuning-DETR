import json

# Define your variables
image_name = "IMG_8579_jpg"
boxes = [
    [
        151.28018188476562,
        424.45782470703125,
        183.20631408691406,
        514.0
    ]
]
labels = [2]
scores = [0.2681429088115692]

# Create a dictionary with the desired format
data = {
    image_name: {
        "boxes": boxes,
        "labels": labels,
        "scores": scores
    }
}

# Convert the data to JSON format
json_data = json.dumps(data, indent=4)

# Save the JSON data to a file
with open("output.json", "w") as json_file:
    json_file.write(json_data)

print("JSON data has been saved to output.json")
