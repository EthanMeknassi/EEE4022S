from PIL import Image
from datetime import datetime

# Define the path to the folder containing the images
folder_path = './Grabbing/'

# Define the number of image pairs you want to process
num_pairs = 149  # Change this to the number of pairs you have

# Define the crop coordinates and dimensions for each image
crop_coords = {
    'rtm': (105, 49, 709, 583),
    'spec': (105, 49, 688, 583),
}

for i in range(1, num_pairs + 1):
    # Load the rtm and spec images
    rtm_image = Image.open(f'{folder_path}rtm_{i}.png')
    spec_image = Image.open(f'{folder_path}spec_{i}.png')
    
    # Crop the images
    rtm_image = rtm_image.crop(crop_coords['rtm'])
    spec_image = spec_image.crop(crop_coords['spec'])
    
    # Calculate the width of the combined image
    total_width = rtm_image.width + spec_image.width
    
    # Create a new blank image with the combined width and the maximum height
    min_height = min(rtm_image.height, spec_image.height)
    combined_image = Image.new('RGB', (total_width, min_height))
    
    # Paste the rtm and spec images side by side
    combined_image.paste(rtm_image, (0, 0))
    combined_image.paste(spec_image, (rtm_image.width, 0))
    
    # Generate a unique filename based on the current date and time
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f'./Grabbing/combined_{i}.jpg'
    
    # Save the combined image with the unique filename
    combined_image.save(output_filename)
    
    # Display the combined image (optional)
    #combined_image.show()
    
    print(f"Saved combined image {i} as {output_filename}")