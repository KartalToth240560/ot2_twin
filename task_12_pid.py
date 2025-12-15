import os
from sim_class import Simulation

from task_5.inference_single import *
from task_5.models.simple_unet import f1
from tensorflow.keras.models import load_model

from bottom_positons import get_bottom

# ... [End of your Simulation class] ...

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    
    patch_size = 256

    example_file_name = f'../deliverables/kartaltoth_240560_unet_model_{patch_size}px.h5'
    model = load_model(example_file_name, custom_objects={"f1": f1})
    
    # 1. Initialize the simulation
    # render=False is faster if you just want to check the texture logic
    sim = Simulation(num_agents=1, render=False)

    # 2. Get the image path
    image_path = sim.get_plate_image()
    print(f"Loading image from: {image_path}")

    # 3. Display using Matplotlib
    try:
        # Read the image
        img = mpimg.imread(image_path)
        
        predicted_mask, preprocess_info = inference(image_path, model, patch_size, threshold=0.3)

        root_bottoms = get_bottom(predicted_mask, preprocess_info)
        
        plt.imshow(predicted_mask)
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found. Check your 'textures' folder.")
    except Exception as e:
        print(f"An error occurred while displaying the image: {e}")

    # 4. Finish the env
    sim.close()
    print("Environment closed.")