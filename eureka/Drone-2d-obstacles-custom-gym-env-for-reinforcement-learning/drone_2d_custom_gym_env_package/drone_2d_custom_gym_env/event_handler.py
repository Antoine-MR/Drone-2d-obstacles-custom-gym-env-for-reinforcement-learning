import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import sys

def pygame_events(space, myenv, change_target):
    try:
        # Check if pygame display is initialized
        if not pygame.get_init() or not pygame.display.get_init():
            return
        
        # Check if there's a display surface
        screen = pygame.display.get_surface()
        if screen is None:
            return
            
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if change_target == True and event.type == pygame.MOUSEBUTTONUP:
                x, y = pygame.mouse.get_pos()
                myenv.change_target_point(x, 800-y)
                
    except (pygame.error, SystemError) as e:
        # Handle pygame errors gracefully
        print(f"Warning: pygame event handling error: {e}")
        return
