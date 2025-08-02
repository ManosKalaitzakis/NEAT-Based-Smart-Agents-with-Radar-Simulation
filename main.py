import math, random, pygame, neat, sys

# Window dimensions and colors
WIDTH, HEIGHT = 1920, 1080
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (237, 28, 36)
GREEN = (181, 230, 29)
BLUE = (0, 162, 232)

# Constants
STILL_LIMIT = 10
FONT_SIZE = 40
current_generation = 0

# Reporter to show fitness in console
class LiveScoreReporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        avg_fitness = sum(g.fitness for g in population.values()) / len(population)
        print(f"Live Score - Best: {best_genome.fitness:.2f}, Avg: {avg_fitness:.2f}")

# Agent class
class Circle:
    def __init__(self, color, start_pos, map_surface):
        self.color = color
        self.pos = list(start_pos)
        self.map_surface = map_surface
        self.alive = True
        self.finished = False
        self.still_frames = 0
        self.distance_moved = 0
        self.last_pos = list(start_pos)
        self.direction = 0
        self.time_alive = 0
        self.turns_count = 0
        self.turns_without_moving = 0
        self.stuck_counter = 0
        self.stuck_pos = list(start_pos)

    def move(self, outputs):
        if not self.alive or self.finished:
            return

        self.time_alive += 1
        speed = 13
        turn_angle = 45
        moved = False

        forward = outputs[0] > 0.5
        left = outputs[1] > 0.5
        right = outputs[2] > 0.5
        brake = outputs[3] > 0.5

        if left:
            self.direction = (self.direction - turn_angle) % 360
            self.turns_count += 1
        if right:
            self.direction = (self.direction + turn_angle) % 360
            self.turns_count += 1

        move_step = speed if forward and not brake else speed * 0.4 if (left or right) else 0

        if move_step > 0:
            rad = math.radians(self.direction)
            new_x = self.pos[0] + move_step * math.cos(rad)
            new_y = self.pos[1] + move_step * math.sin(rad)
            new_pos = (new_x, new_y)
            moved = True
        else:
            new_pos = self.pos

        self.turns_without_moving = 0 if moved else self.turns_without_moving + 1
        if self.turns_without_moving > 30:
            self.alive = False
            return

        if (int(new_pos[0]), int(new_pos[1])) == (int(self.last_pos[0]), int(self.last_pos[1])):
            self.alive = False
            return

        x, y = max(0, min(WIDTH - 1, new_pos[0])), max(0, min(HEIGHT - 1, new_pos[1]))
        pixel = self.map_surface.get_at((int(x), int(y)))[:3]

        if pixel == WHITE:
            self.alive = False
            return

        dist = math.dist(self.pos, (x, y))
        if dist < 0.5:
            self.still_frames += 1
            if self.still_frames > STILL_LIMIT:
                self.alive = False
        else:
            self.still_frames = 0
            self.distance_moved += dist
            self.pos = [x, y]

        if pixel == self.color:
            self.finished = True
            return

        self.last_pos = list(self.pos)

        if abs(self.pos[0] - self.stuck_pos[0]) < 200 and abs(self.pos[1] - self.stuck_pos[1]) < 110:
            self.stuck_counter += 1
            if self.stuck_counter > 60:
                self.alive = False
                return
        else:
            self.stuck_counter = 0
            self.stuck_pos = list(self.pos)

        if self.time_alive > 1360:
            self.alive = False
            return

    def radar(self, angle_offset):
        length = 0
        x, y = self.pos
        angle = math.radians((self.direction + angle_offset) % 360)
        while length < 100:
            check_x = int(x + length * math.cos(angle))
            check_y = int(y + length * math.sin(angle))
            if 0 <= check_x < WIDTH and 0 <= check_y < HEIGHT:
                pixel = self.map_surface.get_at((check_x, check_y))[:3]
                if pixel == WHITE:
                    break
            else:
                break
            length += 2
        return length / 100

    def get_data(self):
        front = self.radar(0)
        left = self.radar(-45)
        right = self.radar(45)
        dir_norm = self.direction / 360.0
        r, g, b = [c / 255 for c in self.color]
        return [front, left, right, dir_norm, r, g, b]

    def get_reward(self):
        if self.finished:
            return 50000 + self.distance_moved * 10 + 100 * self.time_alive

        if not self.alive:
            penalty = max(0, 1 - (self.turns_without_moving / 0.2))
            return -100 + self.distance_moved * 1.0 + 0.05 * self.time_alive + 0.3 * self.turns_count * penalty

        penalty = max(0, 1 - (self.turns_without_moving / 1))
        return self.distance_moved * 1.0 + 0.05 * self.time_alive + 0.3 * self.turns_count * penalty

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.pos[0]), int(self.pos[1])), 8)
        end_x = int(self.pos[0] + 15 * math.cos(math.radians(self.direction)))
        end_y = int(self.pos[1] + 15 * math.sin(math.radians(self.direction)))
        pygame.draw.line(screen, BLACK, self.pos, (end_x, end_y), 2)

# Simulation loop
def run_simulation(genomes, config):
    global current_generation
    current_generation += 1

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", FONT_SIZE)
    map_surface = pygame.image.load('map6.png').convert()

    nets, circles = [], []
    start_positions = [(830, 920)] * len(genomes)
    for i, (gid, genome) in enumerate(genomes):
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        genome.fitness = 0
        color = random.choice([RED, GREEN, BLUE])
        circles.append(Circle(color, start_positions[i], map_surface))

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        alive_count = sum(c.alive and not c.finished for c in circles)
        if alive_count < 1:
            break

        screen.blit(map_surface, (0, 0))
        for i, circle in enumerate(circles):
            if circle.alive and not circle.finished:
                inputs = circle.get_data()
                outputs = nets[i].activate(inputs)
                circle.move(outputs)
            circle.draw(screen)

        # Assign fitness to genomes
        rewards = [c.get_reward() for c in circles]
        finished_count = sum(c.finished for c in circles)
        group_bonus = finished_count * 500
        for i, circle in enumerate(circles):
            base_reward = rewards[i]
            genomes[i][1].fitness = base_reward + group_bonus if circle.finished else base_reward

        # Draw HUD info
        rewards = [g[1].fitness for g in genomes]
        best_fitness = max(rewards)
        avg_fitness = sum(rewards) / len(rewards)
        info_lines = [
            f"Generation: {current_generation}",
            f"Alive: {alive_count}",
            f"Finished: {finished_count}",
            f"Population: {len(circles)}",
            f"Best Fitness: {best_fitness:.0f}",
            f"Avg Fitness: {avg_fitness:.0f}"
        ]
        pygame.draw.rect(screen, WHITE, (5, 5, 220, FONT_SIZE * len(info_lines) + 20))
        for idx, line in enumerate(info_lines):
            screen.blit(font.render(line, True, BLACK), (10, 10 + idx * (FONT_SIZE + 5)))

        pygame.display.flip()

        # Screenshot feature
        keys = pygame.key.get_pressed()
        if keys[pygame.K_s]:
            pygame.image.save(screen, "radar_screenshot.png")

        clock.tick(60)

# Entry point
if __name__ == "__main__":
    config_path = "./config.txt"
    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(LiveScoreReporter())

    population.run(run_simulation, 100)
