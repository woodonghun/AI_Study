import os
import cv2
import json

"""
    coco format dataset 의 image 에 bounding box annotate labeling
"""
# Read the JSON file
with open(r'C:\Object_Detection\data\original\training_data\training_data\quadrant_enumeration\new.json', 'r') as f:
    data = json.load(f)

# Folder path containing the images
folder_path = r'C:\Object_Detection\data\original\training_data\training_data\quadrant_enumeration\xrays/'
save_path = r'C:\Object_Detection\data\original\training_data\training_data\quadrant_enumeration\new/'

# Iterate through each image file in the folder
for image_file in os.listdir(folder_path):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        # Read the image
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        image_id = ''
        # Get the image ID
        for i in data['images']:
            if i['file_name'] == image_file:
                image_id = i['id']

        # Iterate through each annotation in the JSON file
        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                print(annotation['image_id'], image_id)
                bbox = annotation['bbox']
                x, y, width, height = map(int, bbox)
                label = annotation['category_id']
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the image with bounding boxes and labels
        output_path = os.path.join(save_path, 'annotated_' + image_file)
        cv2.imwrite(output_path, image)

print("Annotation completed for all images.")
