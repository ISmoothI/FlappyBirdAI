"""
This is the main version with the AI implemented and a GUI with pygame
Note: sound effects are from Mario, from Gaming Sound FX on youtube, https://www.youtube.com/channel/UCi-xN4ZB6e-0JcXzvBEomlw
"""

import pygame
import neat # documentation: https://neat-python.readthedocs.io/en/latest/config_file.html
import time
import os 
import random 
import pickle # to save our model

icon = pygame.image.load(os.path.join("imgs", "bird1.png"))
pygame.display.set_caption("Flappy Bird with AI")
pygame.display.set_icon(icon)

pygame.mixer.init(44100, -16, 2, 32) # initialize pygame mixer for game sound effects

pygame.font.init() # initialize pygame fonts

# size of the window. Note that all caps in python signals a constant
WIN_WIDTH = 500
WIN_HEIGHT = 800
# create window 
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

# generation of birds
gen = 0

# pygame.transform.scale2x() makes an image 2x larger
# pygame.image.load() loads an image
# os.path.join() joins together directories/files automatically (used because it works with cross-platform)
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

# set bird images
REGULAR_BIRD = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]

# actual bird image used in game - CUSTOMIZE THIS 
BIRD_IMGS = REGULAR_BIRD

FLAPPY_BIRD_FONT = pygame.font.Font(os.path.join("fonts", "flappybirdy", "FlappyBirdy.ttf"), 90)
FLAPPY_BIRD_FONT_SMALL = pygame.font.Font(os.path.join("fonts", "flappybirdy", "FlappyBirdy.ttf"), 50)

STAT_FONT_BOLD = pygame.font.Font(os.path.join("fonts", "Roboto", "Roboto-Bold.ttf"), 50)
STAT_FONT_SMALL = pygame.font.Font(os.path.join("fonts", "Roboto", "Roboto-Regular.ttf"), 25)
EXTRA_SMALL_FONT = pygame.font.Font(os.path.join("fonts", "Roboto", "Roboto-Regular.ttf"), 15)

# sound effects, use with effect.play()
JUMP_EFFECT = pygame.mixer.Sound(os.path.join("music", "jump.wav"))
JUMP_EFFECT.set_volume(0.05)

DEATH_EFFECT = pygame.mixer.Sound(os.path.join("music", "death.wav"))
DEATH_EFFECT.set_volume(0.15)

POINT_EFFECT = pygame.mixer.Sound(os.path.join("music", "point.wav"))
POINT_EFFECT.set_volume(0.15)

# ----- we'll create a class for each of the main objects:

class Bird:
    """
    The bird class represents the flappy bird.
    """
    global BIRD_IMGS
    MAX_ROTATION = 25 # how much bird rotates in flaps
    ROT_VEL = 20 # rotation velocity, how much we rotate/frame
    ANIMATION_TIME = 5 # how long to show each bird animation

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0 # starts with 0 tilt
        self.tick_count = 0
        self.vel = 0 # velocity starts at 0
        self.height = self.y
        self.img_count = 0 # what image we're currently showing for animations
        self.img = BIRD_IMGS[0] # start with first img

    def jump(self):
        self.vel = -10.5 # this means that when jumping, move 10.5px up, since (0,0) is top left corner
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        # physics equation sigh... calculates displacement
        #  x = v_x * t + a * t^2 (acceleration is constant at 1.5 in this case)
        displacement = self.vel*self.tick_count + 1.5 * self.tick_count**2

        # don't move too fast
        if displacement >= 16:
            displacement = 16

        # if it's already jumping, make it jump higher
        if displacement < 0:
            displacement -= 2

        # add the displacement to it
        self.y += displacement

        # if we're still moving up, then we want to make sure we don't tilt too much
        if (displacement < 0 or self.y < self.height + 50):
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            # we want to tilt it all the way down when falling
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1 # track how many times we've shown an img

        # this gives the animation- looks like the bird flaps
        if self.img_count < self.ANIMATION_TIME:
            self.img = BIRD_IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = BIRD_IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = BIRD_IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = BIRD_IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = BIRD_IMGS[0]
            self.img_count = 0

        # if it's already looking down, then just make it look down
        if self.tilt <= -80:
            self.img = BIRD_IMGS[1]
            self.img_count = self.ANIMATION_TIME*2 # skips to the correct frame

        # rotates the image according to tilt
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        # this just makes sure that we rotate the image at its center instead of the top left (default)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe: 
    """
    Represents a pipe object.
    """
    GAP = 200 # gap between pipes
    VEL = 5

    def __init__(self, x): 
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0

        # the PIPE_TOP stores the flipped image of the original pipe
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()
    
    def set_height(self): 
        self.height = random.randrange(50, 450) # generate random height 
        # calculate the heights of where the pipes go
        self.top = self.height - self.PIPE_TOP.get_height() 
        self.bottom = self.height + self.GAP
    
    def move(self): 
        self.x -= self.VEL
    
    def draw(self, win): 
        # .blit to draw the pipe to the screen
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    # this method uses masks in order to have pixel perfect collisions
    # masks are in 2d arrays, and the method checks if actual pixels (not invisible parts of a png image) are overlapping
    def collide(self, bird, win):
        bird_mask = bird.get_mask() 
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        # calculate how far away the corners are away from each other
        top_offset = (self.x - bird.x, self.top - round(bird.y)) # the rounding is just so there's no decimals
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        # this function from pygame returns none if there is no collision
        b_point = bird_mask.overlap(bottom_mask, bottom_offset) 
        t_point = bird_mask.overlap(top_mask,top_offset)

        # if they are colliding, something will be there, so return true for the collision
        if b_point or t_point:
            return True

        return False
 
class Base: 
    """
    Represents the moving floor of the game.
    """
    VEL = 5 # same as pipe, how fast it moves
    WIDTH = BASE_IMG.get_width() 
    IMG = BASE_IMG

    # what we are doing is essentially drawing two different bases
    # we can keep making the background look like it's moving to the left by keeping track of the two imgs with x1 and x2
    
    # if the blocks look like this, where the screen is in []:    [(----)](----)  
    # then once the first block moves all the way to the left like this:    (----)[(----)]
    # we will move the first block to the end, thus creating a cycle

    def __init__(self, y): 
        self.y = y
        self.x1 = 0 
        self.x2 = self.WIDTH

    def move(self): 
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        # if the first image is off the sceen and < 0, then we want to move that to the very right 
        if (self.x1 + self.WIDTH < 0): 
            self.x1 = self.x2 + self.WIDTH
        
        # same for second image once we start cycling
        if (self.x2 + self.WIDTH < 0): 
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win): 
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

class Button:
    """
    Creates a button that can be drawn to the screen.
    init((color), fontSize, x, y, width, height, text="")
    """
    def __init__(self, color, fontSize, x, y, width, height, text=""):
        self.color = color
        self.fontSize = fontSize
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def draw(self, win, outline=None, outline_size=2):
        #Call this method to draw the button on the screen
        if outline:
            # if the user gives us an outline and an outline size, then add an outline
            pygame.draw.rect(win, outline, (self.x - outline_size, self.y - outline_size, self.width + (2 * outline_size), self.height + (2 * outline_size)), 0)
        
        if self.color != None: 
            pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height), 0)
        
        if self.text != '':
            font = pygame.font.Font(os.path.join("fonts", "Roboto", "Roboto-Regular.ttf"), self.fontSize)
            text = font.render(self.text, 1, (0,0,0))
            # draw the text to the screen, centering it according to the box
            win.blit(text, (self.x + (self.width/2 - text.get_width()/2), self.y + (self.height/2 - text.get_height()/2)))

    def isOver(self, pos):
        # pos is the mouse position or a tuple of (x,y) coordinates
        if pos[0] > self.x and pos[0] < self.x + self.width:
            if pos[1] > self.y and pos[1] < self.y + self.height:
                return True
        # pos is going to be the mouse position returned from pygame.mouse.get_pos()    
        return False

# draws the main game window of the game
def draw_window(win, birds, pipes, base, score, exit_button=None, gen=None, threshold_msg=None): 
    """
    Draws everything to the window
    """ 
    # blit is a method from pygame to draw something (usually for images) on the screen
    win.blit(BG_IMG, (0,0)) 
    
    for pipe in pipes: 
        pipe.draw(win) # draw pipes
    
    # set the text to be equal to the score
    text = STAT_FONT_BOLD.render(str(score), 1, (255, 255, 255))
    # draw the score- this just checks to make sure that the text is never off the screen 
    win.blit(text, (WIN_WIDTH/2 - text.get_width()/2, 125)) 
    
    if gen is not None:
        # draw the generation
        text = STAT_FONT_SMALL.render("Generation: " + str(gen), 1, (255, 255, 255))
        win.blit(text, (10, 10)) 

        # draw number of birds alive
        text = STAT_FONT_SMALL.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
        win.blit(text, (10, 35)) 

    base.draw(win) # draw base

    if exit_button is not None:
        exit_button.draw(win) # draw exit button 

    if threshold_msg is not None: 
        win.blit(threshold_msg, (WIN_WIDTH - threshold_msg.get_width() - 5, 55))

    for bird in birds:
        bird.draw(win) # draw every bird in list

    pygame.display.update() # refreshes the window

# main loop that trains a new neural network
def main(genomes, config):
    global gen # declares a global variable 

    gen += 1 # add one to generation

    # nets[i] corresponds with ge[i] and birds [i]
    nets = [] # neural networks
    ge = [] # genomes
    birds = []  # change to a list to work with multiple birds

    # keep track if a bird has met the threshold yet or not
    threshold_met = False
    threshold_msg = None

    # genome looks like (genome_id, genome_object)- we only want the g object
    for _, g in genomes: 
        net = neat.nn.FeedForwardNetwork.create(g, config) # set up neural network
        nets.append(net) # add it to the list
        birds.append(Bird(230, 350)) # add the bird
        g.fitness = 0
        ge.append(g) # add the genome to the list

    base = Base(730) # the height is 800, so put the base at 730 since it's 70px tall
    pipes = [Pipe(600)]
    clock = pygame.time.Clock() # create a clock in order to keep track of how many ticks per second we do
    score = 0 # keep track of the user score (+1 every passed pipe)

    exit_button = Button((197, 235, 207), 25, WIN_WIDTH - 100, 0, 100, 50, "Exit")

    run = True
    while run:
        clock.tick(30) # means we do 30 ticks per second at max 
        for event in pygame.event.get(): # just keeps track of user input
            if event.type == pygame.QUIT:
                run = False # quit the game
                pygame.quit()
                quit()

            pos = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONDOWN: 
                if exit_button.isOver(pos) and threshold_met: 
                    run = False
                else: 
                    threshold_msg = EXTRA_SMALL_FONT.render("Wait for fitness to be met", 1, (255, 255, 255))
                    
        # if the threshold message is already something (saying "wait for fitness to be met")
        # then we let the user know when it's met so they know when they can exit
        if threshold_msg is not None and threshold_met:  
            threshold_msg = EXTRA_SMALL_FONT.render("Fitness threshold met", 1, (255, 255, 255))

        #bird.move() # move the bird
        pipe_ind = 0 # keep track of what pipe we're looking at
        if len(birds) > 0: 
            # if we passed a pipe, then change the pipe we're looking at to the next one
            if (len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width()):
                 pipe_ind = 1
        else:
            # if there's no birds left, quit current round of game 
            run = False
            break

        for x, bird in enumerate(birds): 
            bird.move()
            # add fitness to the bird if it survives time
            ge[x].fitness += 0.1 # we want to encourage the bird to not kill itself
            
            # if we hit good fitness, allow the user to exit
            if(ge[x].fitness > 100):
                threshold_met = True

            # this function will return something from -1 to 1 due to our tanh activation function
            # we pass in the bird's spot, and the spot of the bottom pipe's height and the top pipe's bottom
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            # output is a list of output neurons- we only have one, so use index 0 
            if (output[0] > 0.5): 
                bird.jump()

        rem = [] # list of pipes to remove
        add_pipe = False
        for pipe in pipes: 
            for x, bird in enumerate(birds): 
                if pipe.collide(bird, win): 
                    ge[x].fitness -= 1 # it's bad for the bird to hit the pipe
                    birds.pop(x) # get rid of bird
                    nets.pop(x) # also get rid of network at same spot
                    ge.pop(x) # get rid of the ge associated with the bird
                
                # if we passed the pipe
                if(not pipe.passed and pipe.x < bird.x): 
                    pipe.passed = True
                    add_pipe = True

            # if the pipe is off of the screen, then we need to add new pipes
            if (pipe.x + pipe.PIPE_TOP.get_width() < 0):
                rem.append(pipe) 
            
            pipe.move()

        # then we have to add a score and a new pipe
        if add_pipe: 
            score += 1
            for g in ge: 
                g.fitness += 5 # it's good for birds to pass the pipe
            pipes.append(Pipe(600)) # add new pipe

        #remove everything in the array of pipes to be removed 
        for r in rem: 
            pipes.remove(r)

        # check collision between bird and floor
        for x, bird in enumerate(birds): 
            if (bird.y + bird.img.get_height() >= 730 or bird.y < 0):
                birds.pop(x) 
                nets.pop(x) 
                ge.pop(x) 

        base.move() # move the base
        draw_window(win, birds, pipes, base, score, exit_button, gen, threshold_msg)

        if score >= 50:
            # open file, "wb" = "write binary"
            with open("best.pickle", "wb") as f:
                pickle.dump(nets[0], f)

# main loop with best AI 
def run_best_AI():
    bird = Bird(230, 350) # there will only be one bird
    
    with open("best.pickle", "rb") as f:
        model = pickle.load(f) # model will be saved here

    base = Base(730) # the height is 800, so put the base at 730 since it's 70px tall
    pipes = [Pipe(600)]
    clock = pygame.time.Clock() # create a clock in order to keep track of how many ticks per second we do
    score = 0 # keep track of the user score (+1 every passed pipe)

    exit_button = Button((197, 235, 207), 25, WIN_WIDTH - 100, 0, 100, 50, "Exit")

    run = True
    while run:
        clock.tick(30) # means we do 30 ticks per second at max 
        for event in pygame.event.get(): # just keeps track of user input
            if event.type == pygame.QUIT:
                run = False # quit the game
                pygame.quit()
                quit()

            pos = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONDOWN: 
                if exit_button.isOver(pos): 
                    run = False

        pipe_ind = 0 # keep track of what pipe we're looking at
        # if we passed a pipe, then change the pipe we're looking at to the next one
        if (len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width()):
                pipe_ind = 1

        bird.move()
        # use pretrained model
        output = model.activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
        # output is a list of output neurons- we only have one, so use index 0 
        if (output[0] > 0.5): 
            JUMP_EFFECT.play()
            bird.jump()

        rem = [] # list of pipes to remove
        add_pipe = False
        for pipe in pipes: 
            if pipe.collide(bird, win): 
                run = False 
                DEATH_EFFECT.play()
                break
                
            # if we passed the pipe
            if(not pipe.passed and pipe.x < bird.x): 
                pipe.passed = True
                add_pipe = True

            # if the pipe is off of the screen, then we need to add new pipes
            if (pipe.x + pipe.PIPE_TOP.get_width() < 0):
                rem.append(pipe) 
            
            pipe.move()

        # then we have to add a score and a new pipe
        if add_pipe: 
            score += 1
            POINT_EFFECT.play()
            pipes.append(Pipe(600)) # add new pipe

        #remove everything in the array of pipes to be removed 
        for r in rem: 
            pipes.remove(r)

        # check collision between bird and floor
        if (bird.y + bird.img.get_height() >= 730 or bird.y < 0):
            run = False
            DEATH_EFFECT.play()
            break

        base.move() # move the base
        draw_window(win, [bird], pipes, base, score, exit_button)

# main loop with manual play
def run_normal_mode():
    bird = Bird(230, 350) # there will only be one bird

    base = Base(730) # the height is 800, so put the base at 730 since it's 70px tall
    pipes = [Pipe(600)]
    clock = pygame.time.Clock() # create a clock in order to keep track of how many ticks per second we do
    score = 0 # keep track of the user score (+1 every passed pipe)
    
    run = True
    while run:
        clock.tick(30) # means we do 30 ticks per second at max 
        for event in pygame.event.get(): # just keeps track of user input
            if event.type == pygame.QUIT:
                run = False # quit the game
                pygame.quit()
                quit()

        pipe_ind = 0 # keep track of what pipe we're looking at
        # if we passed a pipe, then change the pipe we're looking at to the next one
        if (len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width()):
                pipe_ind = 1

        bird.move()
        
        # see what the user wants to do:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            bird.jump()
            JUMP_EFFECT.play()

        rem = [] # list of pipes to remove
        add_pipe = False
        for pipe in pipes: 
            if pipe.collide(bird, win): 
                run = False 
                DEATH_EFFECT.play()
                break
                
            # if we passed the pipe
            if(not pipe.passed and pipe.x < bird.x): 
                pipe.passed = True
                add_pipe = True

            # if the pipe is off of the screen, then we need to add new pipes
            if (pipe.x + pipe.PIPE_TOP.get_width() < 0):
                rem.append(pipe) 
            
            pipe.move()

        # then we have to add a score and a new pipe
        if add_pipe: 
            score += 1
            POINT_EFFECT.play()
            pipes.append(Pipe(600)) # add new pipe

        #remove everything in the array of pipes to be removed 
        for r in rem: 
            pipes.remove(r)

        # check collision between bird and floor
        if (bird.y + bird.img.get_height() >= 730 or bird.y < 0):
            run = False
            DEATH_EFFECT.play()
            break

        base.move() # move the base
        draw_window(win, [bird], pipes, base, score)

# runs the NEAT AI
def runAI(config_path): 
    # this is a method from the NEAT module to configure what our AI will be like
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_path)
    # generate population
    p = neat.Population(config)

    # these give us output in the console
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    # provide fitness function and how many times this runs
    winner = p.run(main, 50)

# redraws the main gui 
def draw_main_gui(*argv):
    # fill the window with white and then draw the background
    win.fill((255, 255, 255))
    win.blit(BG_IMG, (0,0)) 

    # create title with flappy bird font
    title = FLAPPY_BIRD_FONT.render("Flappy Bird", 1, (0,0,0))
    subtitle = FLAPPY_BIRD_FONT_SMALL.render("with AI", 1, (0,0,0))

    # draw the text to the screen
    win.blit(title, (WIN_WIDTH/2 - title.get_width()/2, 100))
    win.blit(subtitle, (WIN_WIDTH/2 - subtitle.get_width()/2, 175))

    # draw everything given to us
    for x in argv: 
        x.draw(win)

# run the settings gui with the skins
def run_settings_gui():
    global BIRD_IMGS # we will be updating the bird images

    # fill the window with white and then draw the background
    win.fill((255, 255, 255))
    win.blit(BG_IMG, (0,0)) 

    # create title with flappy bird font
    title = FLAPPY_BIRD_FONT.render("Skins", 1, (0,0,0))

    # draw the text to the screen
    win.blit(title, (WIN_WIDTH/2 - title.get_width()/2, 100))

    button_width = 175
    buttonX = 250 - button_width/2
    buttonY = 250 # where first button starts
    button_gap = 80 # space between each button

    normal_button = Button((197, 235, 207), 25, buttonX, buttonY, button_width, 60, "Normal Bird")

    normal_button.draw(win)

    run = True
    clock = pygame.time.Clock()
    while run: 
        clock.tick(30)
        
        pygame.display.update() # IMPORTANT

        # grab the events from pygame
        for event in pygame.event.get(): 
            # position of the mouse
            pos = pygame.mouse.get_pos()

            if event.type == pygame.QUIT:
                run = False # quit the game
                pygame.quit()
                quit()

            # set bird images accordingly and exit the menu
            if event.type == pygame.MOUSEBUTTONDOWN: 
                if normal_button.isOver(pos):
                    BIRD_IMGS = REGULAR_BIRD
                    run = False

# runs the main gui of the game
def run_gui(): 
    run = True
    centerX = 250 # center of screen
    button_width = 200 # how wide the button is
    bird = Bird(82, 50) # bird flapping in the corner
    buttonY = 250 # where the first button starts
    button_gap = 90 # space between each button
    
    # create buttons - init((color), fontSize, x, y, width, height, text="")
    noAI_button = Button((197, 235, 207), 25, centerX - (button_width/2), buttonY, button_width, 60, "Manual Play")
    trainAI_button = Button((197, 235, 207), 25, centerX - (button_width/2), buttonY + button_gap, button_width, 60, "Train AI")
    bestAI_button = Button((197, 235, 207), 25, centerX - (button_width/2), buttonY + 2 * button_gap, button_width, 60, "Run best AI")
    settings_button = Button((197, 235, 207), 25, centerX - (button_width/2), buttonY + 3 * button_gap, button_width, 60, "Skins")

    clock = pygame.time.Clock()

    while run: 
        # reset the generation of the birds
        global gen # global keyword so we can modify the global variable
        gen = 0

        clock.tick(30) # limit how many ticks there are
        
        draw_main_gui(bird, noAI_button, trainAI_button, bestAI_button, settings_button)
        pygame.display.update()

        # grab the events from pygame
        for event in pygame.event.get(): 
            # position of the mouse
            pos = pygame.mouse.get_pos()

            if event.type == pygame.QUIT:
                run = False # quit the game
                pygame.quit()
                quit()

            if event.type == pygame.MOUSEBUTTONDOWN: 
                if trainAI_button.isOver(pos):
                    # this is a common thing in python: if the current file is the "__main__", that means
                    # it's the main file running. Then, we can use this to manipulate the path to the .txt file
                    if __name__ == '__main__':
                        local_dir = os.path.dirname(__file__) # this is the directory we're in
                        config_path = os.path.join(local_dir, 'config-feedforward.txt') # join current dir with the file
                        runAI(config_path)
                if bestAI_button.isOver(pos): 
                    run_best_AI()
                if noAI_button.isOver(pos): 
                    run_normal_mode()
                if settings_button.isOver(pos):
                    run_settings_gui()
                    
run_gui() # run the main gui
