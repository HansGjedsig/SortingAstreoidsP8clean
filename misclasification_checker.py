import os
import pandas as pd
import pygame
import numpy as np
from pygame.locals import QUIT, KEYDOWN, K_LEFT, K_RIGHT, K_UP

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
TXTCOL = (0,255,0)
data = pd.read_csv('misclasifications.csv')



def main(data):

    # Set up pygame
    pygame.init()

    # Set the screen dimensions based on your images
    screen_width = 500
    screen_height = 500
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Image Viewer")

    # Specify the folder paths
    left_folder = 'training_data/bad_thumbs'
    right_folder = 'training_data/good_thumbs'
    up_folder = 'training_data/hmmm'

    
    # Get a list of image files in the source folder
    image_files = data["path"].copy()

    # Initialize variables
    current_index = 0
    index_list = []
    running = True
    # Load font for tooltip
    font = pygame.font.Font(None, 24)

    while running:
        # Load the current image
        image = pygame.image.load(image_files[current_index])
        # Resize the image to fit the screen
        image = pygame.transform.scale(image, (screen_width, screen_height))

        # Display the image
        screen.blit(image, (0, 0))
        
        # Set the window caption to the file name
        pygame.display.set_caption(image_files[current_index])
        
        # Render tooltip text
        tooltip_left = font.render("Left: Move to 'bad_thumbs'", True, TXTCOL)
        tooltip_right = font.render("Right: Move to 'good_thumbs'", True, TXTCOL)
        tooltip_up = font.render("Classified as "+str(data["prediction"][current_index])+", Assigned as "+str(data["Assigned"][current_index]), True, TXTCOL)
        tooltip_count = font.render(str(current_index)+"/"+str(len(image_files)), True, TXTCOL)
        tooltip_counttogo = font.render("Good% "+str(np.round(data["Good_Prediction%"][current_index],3))+" Bad% "+str(np.round(data["Bad_Prediction%"][current_index],3)), True, TXTCOL)
        

        # Display tooltip text
        screen.blit(tooltip_left, (10, 10))
        screen.blit(tooltip_right, (10, 30))
        screen.blit(tooltip_up, (10, 50))
        screen.blit(tooltip_count, (400, 10))
        screen.blit(tooltip_counttogo, (300, 30))
        
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                data = data.drop(index_list)
                data = data.reset_index(drop=True)
                data.to_csv("misclasifications.csv",index=False)
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    save_image(image_files[current_index], left_folder)
                    index_list.append(current_index)
                if event.key == K_RIGHT:
                    save_image(image_files[current_index], right_folder)
                    index_list.append(current_index)
                if event.key == K_UP:
                    save_image(image_files[current_index], up_folder)
                    index_list.append(current_index)
                    

                # Move to the next image
                current_index += 1
                if current_index == len(image_files):
                    data = data.drop(index_list)
                    data = data.reset_index(drop=True)
                    data.to_csv("misclasifications.csv",index=False)
                    running = False  # End the loop when all images are processed
                

    pygame.quit()

def save_image(image_path, destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Extract the file name from the path
    file_name = os.path.basename(image_path)

    # Construct the destination path
    destination_path = os.path.join(destination_folder, file_name)

    # Copy the file to the destination folder
    os.rename(image_path, destination_path)

if __name__ == "__main__":
    data = pd.read_csv("misclasifications.csv")
    main(data)
