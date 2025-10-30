import pymunk
import pygame
import random
import numpy as np

class Obstacle:
    def __init__(self, x, y, width, height, space):
        """
        Crée un obstacle rectangulaire statique
        """
        self.width = width
        self.height = height
        
        # Créer le body statique
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = x, y
        
        # Créer la forme rectangulaire
        self.shape = pymunk.Poly.create_box(self.body, (width, height))
        self.shape.friction = 0.7
        self.shape.color = pygame.Color(139, 69, 19)  # Couleur marron pour les obstacles
        
        # Ajouter à l'espace physique
        space.add(self.body, self.shape)

class ObstacleManager:
    def __init__(self, space):
        self.space = space
        self.obstacles = []
    
    def add_random_obstacles(self, num_obstacles=3, min_size=40, max_size=80):
        """
        Ajoute des obstacles aléatoires dans l'environnement
        """
        self.clear_obstacles()
        
        for _ in range(num_obstacles):
            # Position aléatoire dans la zone de jeu (éviter les bords)
            x = random.uniform(100, 700)
            y = random.uniform(100, 700)
            
            # Taille aléatoire
            width = random.uniform(min_size, max_size)
            height = random.uniform(min_size, max_size)
            
            obstacle = Obstacle(x, y, width, height, self.space)
            self.obstacles.append(obstacle)
    
    def add_fixed_obstacles(self):
        """
        Ajoute des obstacles à positions fixes pour des tests reproductibles
        Map simple avec un seul obstacle central
        """
        self.clear_obstacles()
        
        # Un seul obstacle au centre (carré)
        obstacle1 = Obstacle(400, 400, 100, 100, self.space)
        self.obstacles.append(obstacle1)
    
    def clear_obstacles(self):
        """
        Supprime tous les obstacles de l'espace
        """
        for obstacle in self.obstacles:
            self.space.remove(obstacle.body, obstacle.shape)
        self.obstacles.clear()
    
    def check_collision_with_drone(self, drone):
        """
        Vérifie si le drone entre en collision avec un obstacle en utilisant les positions et rayons
        """
        # Rayon approximatif du drone (basé sur la largeur)
        drone_radius = drone.drone_radius + 10  # Petit buffer pour la détection
        
        # Positions des différentes parties du drone
        drone_positions = [
            drone.frame_shape.body.position,
            drone.left_motor_shape.body.position,
            drone.right_motor_shape.body.position
        ]
        
        for obstacle in self.obstacles:
            obs_x, obs_y = obstacle.body.position
            obs_half_width = obstacle.width / 2
            obs_half_height = obstacle.height / 2
            
            # Vérifier la collision avec chaque partie du drone
            for drone_pos in drone_positions:
                drone_x, drone_y = drone_pos
                
                # Distance minimale entre le centre du drone et l'obstacle rectangulaire
                dx = max(0, max(obs_x - obs_half_width - drone_x, drone_x - (obs_x + obs_half_width)))
                dy = max(0, max(obs_y - obs_half_height - drone_y, drone_y - (obs_y + obs_half_height)))
                
                # Distance du drone au bord de l'obstacle
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Collision si le drone est trop proche de l'obstacle
                if distance < drone_radius:
                    return True
                    
        return False