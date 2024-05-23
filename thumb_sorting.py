import os
import pygame
from pygame.locals import QUIT, KEYDOWN, K_LEFT, K_RIGHT, K_UP
# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def main():
    # Set up pygame
    pygame.init()

    # Set the screen dimensions based on your images
    screen_width = 500
    screen_height = 500
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Image Viewer")

    # Specify the folder paths
    source_folder = "thumbs"
    left_folder = "bad_thumbs"
    right_folder = "good_thumbs"
    up_folder = "hmmm"

    # Get a list of image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Initialize variables
    current_index = 0
    running = True
    # Load font for tooltip
    font = pygame.font.Font(None, 24)

    while running:
        # Load the current image
        image_path = os.path.join(source_folder, image_files[current_index])
        image = pygame.image.load(image_path)
        # Resize the image to fit the screen
        image = pygame.transform.scale(image, (screen_width, screen_height))

        # Display the image
        screen.blit(image, (0, 0))
        
        # Set the window caption to the file name
        pygame.display.set_caption(image_files[current_index])
        
        # Render tooltip text
        tooltip_left = font.render("Left: Move to 'bad_thumbs'", True, WHITE)
        tooltip_right = font.render("Right: Move to 'good_thumbs'", True, WHITE)
        tooltip_up = font.render("Up: Move to 'hmmm'", True, WHITE)
        tooltip_count = font.render(str(current_index)+"/"+str(len(image_files)), True, WHITE)
        tooltip_counttogo = font.render("ONLY "+str(len(image_files)-current_index)+" TO GO!!", True, WHITE)
        

        # Display tooltip text
        screen.blit(tooltip_left, (10, 10))
        screen.blit(tooltip_right, (10, 30))
        screen.blit(tooltip_up, (10, 50))
        screen.blit(tooltip_count, (400, 10))
        screen.blit(tooltip_counttogo, (340, 30))
        
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    save_image(image_path, left_folder)
                elif event.key == K_RIGHT:
                    save_image(image_path, right_folder)
                elif event.key == K_UP:
                    save_image(image_path, up_folder)

                # Move to the next image
                current_index += 1
                if current_index == len(image_files):
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
    main()
