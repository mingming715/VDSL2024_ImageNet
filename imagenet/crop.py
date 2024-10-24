import os
import xml.etree.ElementTree as ET
from PIL import Image

def crop_and_save(xml_file, output_dir):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get folder and filename from the XML
    folder = root.find('folder').text
    filename = root.find('filename').text
    
    # Construct the image file path
    img_path = os.path.join("/database/PCB_DATASET/images", folder, filename)

    # Open the image file
    img = Image.open(img_path)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each object in the annotation file
    for i, obj in enumerate(root.findall('object')):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Calculate the center of the bounding box
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        
        # Calculate the coordinates of the 256x256 crop
        crop_xmin = max(center_x - 128, 0)
        crop_ymin = max(center_y - 128, 0)
        crop_xmax = min(center_x + 128, img.width)
        crop_ymax = min(center_y + 128, img.height)

        # Crop the image
        crop_img = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

        # Save the cropped image
        crop_img.save(os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_crop_{i}.jpg'))

# Define the directories
annotation_dir = '/database/PCB_DATASET/Annotations/Spurious_copper'
output_base_dir = '/database/PCB_DATASET/crop_images/Spurious_copper'

# Ensure the output base directory exists
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Process each XML file in the annotation directory
for xml_file in os.listdir(annotation_dir):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(annotation_dir, xml_file)
        crop_and_save(xml_path, output_base_dir)
