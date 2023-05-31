import csv
import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import sys
import os
import pandas as pd


# Create an initial chromosome for ES
def create_chromosome(fpga_width, fpga_height, block_sizes):
    chromosome = []
    sorted_blocks = sorted(block_sizes.items(), key=lambda x: x[1], reverse=True)
    for block, area in sorted_blocks:
        width, height = area_to_rectangle(area)
        placed = False
        count = 0
        while not placed and count < 10000:
            count += 1
            x = int(random.randint(0, fpga_width))
            y = int(random.randint(0, fpga_height))
            if check_validity(chromosome, fpga_width, fpga_height, x, y, width, height):
                chromosome.append((block, x, y, width, height))
                placed = True
            if count >= 10000:
                print(f"Block {block} couldn't be placed.")
    # print(chromosome)
    return chromosome


# get the shape (width , height) from the given area
def area_to_rectangle(area):
    width = int(math.sqrt(area))
    height = int(area / width)
    while width * height != area:
        width -= 1
        height = int(area / width)
    return width, height


def new_area_to_rectangle(area):
    if area > 1 and all(area % i != 0 for i in range(2, int(math.sqrt(area)) + 1)):
        # area is prime
        possible_rectangles = [(1, area), (area, 1)]
        if len(set(possible_rectangles)) == 2:
            # there are only two possible rectangle shapes
            if area != 2:
                area += 1
    width = int(math.sqrt(area))
    height = int(area / width)
    while width * height != area:
        width -= 1
        height = int(area / width)
    return width, height


# check function for no overlaps and out of boundaries for chromosome
# def check_validity(chromosome, fpga_width, fpga_height, x, y, width, height):
#     # print('chromosome in check validity',chromose)
#     for block in chromosome:
#         x0, y0 = block[1], block[2]
#         x1 = x0 + block[3]
#         y1 = y0 + block[4]
#         # print(x0,y0,x1,y1)
#         # print(x,y,width,height)
#
#         # check if block is outside the boundaries
#         if x + width > fpga_width or y + height > fpga_height:
#             return False
#
#         # check for overlapping
#         if x < x1 and x + width > x0 and y < y1 and y + height > y0:
#             return False
#     return True

def check_validity(chromosome, fpga_width, fpga_height, x, y, width, height):
    if x < 1 or y < 1 or x + width > fpga_width or y + height > fpga_height:
        return False
    for block in chromosome:
        x0, y0 = block[1], block[2]
        x1 = x0 + block[3]
        y1 = y0 + block[4]
        if x < x1 and x + width > x0 and y < y1 and y + height > y0:
            return False
    return True


# fitness function
def evaluate_fitness(chromosome, fpga_width, fpga_height):
    minx, miny = fpga_width, fpga_height
    maxx, maxy = 0, 0
    total_area = 0
    for block in chromosome:
        x0, y0 = block[1], block[2]
        x1 = x0 + block[3]
        y1 = y0 + block[4]
        minx = min(minx, x0)
        miny = min(miny, y0)
        maxx = max(maxx, x1)
        maxy = max(maxy, y1)
        total_area += block[3] * block[4]
    bounding_rect_area = (maxx - minx) * (maxy - miny)
    unused_area = bounding_rect_area - total_area
    # Penalty factor for unused space within the bounding rectangle
    penalty_factor = 0.1
    # Compute the fitness as the sum of the bounding rectangle area and the penalty for unused space
    fitness = bounding_rect_area + penalty_factor * unused_area
    return fitness


# returns the bounding area of the given chromosome
def bounding_area(chromosome, fpga_width, fpga_height):
    minx, miny = fpga_width, fpga_height
    maxx, maxy = 0, 0
    for block in chromosome:
        x0, y0 = block[1], block[2]
        x1 = x0 + block[3]
        y1 = y0 + block[4]
        # print(x0,y0)
        # print(x1,y1)
        minx = min(minx, x0)
        miny = min(miny, y0)
        maxx = max(maxx, x1)
        maxy = max(maxy, y1)
    # print('maxx',maxx,'minx',minx,'maxy',maxy,'miny',miny)
    mbr = (maxx - minx) * (maxy - miny)
    return mbr


# this function is to visualize the layout(chromosome)
def visualize(chromosome, fpga_width, fpga_height):
    plt.figure(figsize=(10, 10))
    plt.xlim(0, fpga_width)
    plt.ylim(0, fpga_height)
    areas = [block[3] * block[4] for block in chromosome]
    normalized_areas = [(area - min(areas)) / (max(areas) - min(areas)) for area in areas]
    # Create a colormap
    cmap = cm.get_cmap("viridis")
    plt.gca().set_aspect('equal', adjustable='box')
    for i, block in enumerate(chromosome):
        color = cmap(normalized_areas[i])
        plt.gca().add_patch(plt.Rectangle((block[1], block[2]), block[3], block[4], color=color, fill=True))
        plt.gca().text(block[1] + block[3] / 2, block[2] + block[4] / 2, block[0], fontsize=18, color='black',
                       ha='center', va='center')
        # plt.gca().text(block[1] + math.sqrt(block[3])/2, block[2] + math.sqrt(block[4])/2, areas[i], ha='left', va='bottom')
    # Add x and y axis labels
    # plt.xticks(range(fpga_width+1))
    # plt.yticks(range(fpga_height+1))
    plt.xlabel('fpga_width', fontsize=20)
    plt.ylabel('fpga_height', fontsize=20)
    # plt.savefig('final_opti_ES_sample')
    plt.show()


# tournament selection
def tournament_selection(population, fitness, tournament_size):
    # Select a random subset of the population
    subset = random.sample(list(range(len(population))), tournament_size)
    # Choose the individual with the best fitness in the subset
    winner = subset[0]
    for i in subset:
        if fitness[i] < fitness[winner]:
            winner = i
    return population[winner]


# this function gets the x and y positions for the given block in chromosome
max_retries = 1500


def find_new_positions(block, fpga_width, fpga_height, chromosome):
    x, y = block[1], block[2]
    # print(x,y)
    width, height = block[3], block[4]

    # Get the bounding coordinates of the current layout
    x_coords = [rect[1] for rect in chromosome]
    y_coords = [rect[2] for rect in chromosome]
    x0 = min(x_coords)
    y0 = min(y_coords)
    x1 = max([x + rect[3] for x, rect in zip(x_coords, chromosome)])
    y1 = max([y + rect[4] for y, rect in zip(y_coords, chromosome)])

    for i in range(max_retries):
        # Generate a new position for the block randomly within the bounding area
        new_x = int(random.uniform(x0, x1 - width))
        new_y = int(random.uniform(y0, y1 - height))
        if new_x != x or new_y != y:
            # Check if the new block is still inside the FPGA boundaries and does not overlap with other blocks
            if check_validity(chromosome, fpga_width, fpga_height, new_x, new_y, width, height):
                # print(f'accepted new x and y after {i+1} retries')
                return new_x, new_y

    # If the new block is not valid after max_retries, return the original block position
    # print('could not find a valid position after max retries')
    return x, y


# this function gives the new shape which can reduce the total bounding area
def get_best_shape(block, block_area, chromosomes, fpga_width, fpga_height):
    # Find all possible shapes that fit within the block's area
    shapes = []
    for i in range(1, int(math.sqrt(block_area)) + 1):
        if block_area % i == 0:
            shapes.append((i, block_area // i))

    # Compute the bounding box of the current layout
    minx, miny = fpga_width, fpga_height
    maxx, maxy = 0, 0
    for chrom in chromosomes:
        x0, y0 = chrom[1], chrom[2]
        x1, y1 = x0 + chrom[3], y0 + chrom[4]
        minx = min(minx, x0)
        miny = min(miny, y0)
        maxx = max(maxx, x1)
        maxy = max(maxy, y1)

    # Randomly choose positions for the block and compute the resulting bounding box
    best_shape = None
    best_pos = None
    best_area = (maxx - minx) * (maxy - miny)
    for shape in shapes:
        updated_block = (block[0], block[1], block[2], shape[0], shape[1])
        x, y = find_new_positions(updated_block, fpga_width, fpga_height, chromosomes)
        # Compute the bounding box of the layout with the new block
        new_minx = min(minx, x)
        new_miny = min(miny, y)
        new_maxx = max(maxx, x + shape[0])
        new_maxy = max(maxy, y + shape[1])
        new_area = (new_maxx - new_minx) * (new_maxy - new_miny)

        # If the new area is smaller, update the best shape and position
        if new_area < best_area:
            # Check if the new block overlaps with any existing block
            overlap = False
            for chrom in chromosomes:
                if x + shape[0] <= chrom[1] or chrom[1] + chrom[3] <= x or y + shape[1] <= chrom[2] or chrom[2] + chrom[
                    4] <= y:
                    # No overlap
                    pass
                else:
                    # Overlap detected
                    overlap = True
                    break
            if not overlap:
                best_shape = shape
                best_pos = (x, y)
                best_area = new_area

    if best_shape is None:
        # No valid shape found, keep the original shape
        return block
    else:
        # Update the block with the best shape and position
        block = list(block)
        block[1], block[2] = best_pos
        block[3], block[4] = best_shape
        return tuple(block)


def check_validity_new(chromosomes, fpga_width, fpga_height):
    for i in range(len(chromosomes)):
        block = chromosomes[i]
        x0, y0 = block[1], block[2]
        x1, y1 = x0 + block[3], y0 + block[4]
        if x0 < 0 or y0 < 0 or x1 > fpga_width or y1 > fpga_height:
            return False
        for j in range(i + 1, len(chromosomes)):
            other_block = chromosomes[j]
            ox0, oy0 = other_block[1], other_block[2]
            ox1, oy1 = ox0 + other_block[3], oy0 + other_block[4]
            if x0 < ox1 and x1 > ox0 and y0 < oy1 and y1 > oy0:
                return False
    return True


def new_get_best_shape(block, block_area, chromosome, fpga_width, fpga_height):
    # Find all possible shapes that fit within the block's area
    shapes = []
    for i in range(1, int(math.sqrt(block_area)) + 1):
        if block_area % i == 0:
            shapes.append((i, block_area // i))
            if i != block_area // i:
                shapes.append((block_area // i, i))

    # Store the original block and its index in the chromosome
    original_block = list(block)
    block_index = chromosome.index(block)

    # Randomly choose positions for the block and compute the resulting bounding box
    best_shape = None
    best_area = bounding_area(chromosome, fpga_width, fpga_height)  # Initialize with maximum area
    # print(best_area)
    for shape in shapes:
        # print(shape)
        # Update the block's shape in the chromosome
        chromosome[block_index] = (block[0], block[1], block[2], shape[0], shape[1])

        # Check validity and compute the new area
        if check_validity_new(chromosome, fpga_width, fpga_height):
            new_area = bounding_area(chromosome, fpga_width, fpga_height)
            # print(new_area)
            if new_area <= best_area:
                best_shape = shape
                best_area = new_area

        # Revert the chromosome back to the original block
        chromosome[block_index] = tuple(original_block)

    # Update the block's shape with the best shape found, if any
    if best_shape is not None:
        block = list(block)
        block[3], block[4] = best_shape
        block = tuple(block)

    # Return the updated block as a tuple
    return tuple(block)


# rotates the given block and gets new x and y positions
def rotate_the_block(block, chromosomes, fpga_width, fpga_height):
    block_name, x, y, width, height = block
    # rotate the block
    new_width, new_height = height, width
    updated_block = (block_name, x, y, new_width, new_height)
    new_x, new_y = find_new_positions(updated_block, fpga_width, fpga_height, chromosomes)
    if new_x == x and new_y == y:
        new_width, new_height = width, height
        return (block_name, x, y, new_width, new_height)
    else:
        return (block_name, new_x, new_y, new_width, new_height)


# this mutate functions generates new chromosomes by changing the shape of the block or just rotating the block
def mutate(chromosomes, mutation_rate, fpga_width, fpga_height, blocks_dict):
    for i in range(len(chromosomes)):
        if random.uniform(0, 1) < mutation_rate:
            block_name, x, y, width, height = chromosomes[i]
            block_raw_area = blocks_dict[block_name]

            if random.uniform(0, 1) < 0.5:
                updated_block = get_best_shape((block_name, x, y, width, height), block_raw_area, chromosomes,fpga_width, fpga_height)

                chromosomes[i] = (block_name, updated_block[1], updated_block[2], updated_block[3], updated_block[4])
            else:
                new_x, new_y = find_new_positions((block_name, x, y, width, height), fpga_width, fpga_height,
                                                  chromosomes)
                # rotated_block = rotate_the_block((block_name, x, y, width, height), chromosomes, fpga_width,
                #                                  fpga_height)
                chromosomes[i] = (block_name, new_x, new_y, width, height)
                # rotated_block = rotate_the_block((block_name, x, y, width, height), chromosomes, fpga_width,
                #                                  fpga_height)
                # chromosomes[i] = (block_name, rotated_block[1], rotated_block[2], rotated_block[3], rotated_block[4])

    return chromosomes


def evolutionary_strategy(fpga_width, fpga_height, blocks, population_size, mutation_rate, num_generations,
                          tournament_size):
    # Initialize the population
    population = [create_chromosome(fpga_width, fpga_height, blocks) for _ in range(population_size)]
    # Evaluate the fitness of each chromosome
    fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Run the ES for the specified number of generations
    for generation in range(num_generations):

        # Sort the population by their fitness values
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]
        sorted_fitness = sorted(fitness)

        # Keep the top 10% of the best individuals (elites)
        num_elites = int(population_size * 0.1)
        elites = sorted_population[:num_elites]

        # Create the offspring
        offspring = []
        for i in range(population_size - num_elites):
            # Sample a parent from the population using tournament selection
            parent = tournament_selection(population, fitness, tournament_size)
            # print(bounding_area(parent,fpga_width,fpga_height))
            # Create an offspring by mutating the parent
            mutated_offspring = mutate(parent, mutation_rate, fpga_width, fpga_height, blocks)
            offspring.append(mutated_offspring)

        # Add the elites to the offspring
        offspring += elites

        # Replace the population with the offspring
        population = offspring

        # Evaluate the fitness of the offspring
        fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Return the best chromosome
    best_chromosome = population[fitness.index(min(fitness))]

    return best_chromosome


def evolutionary_strategy_timed(fpga_width, fpga_height, blocks, population_size, mutation_rate, num_generations,
                                tournament_size, max_time):
    # Initialize the population
    population = [create_chromosome(fpga_width, fpga_height, blocks) for _ in range(population_size)]
    # Evaluate the fitness of each chromosome
    fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Set the start time
    start_time = time.time()

    # Run the ES for the specified number of generations or until the time limit is reached
    while (time.time() - start_time) < max_time:
        # Sort the population by their fitness values
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]
        sorted_fitness = sorted(fitness)

        # Keep the top 10% of the best individuals (elites)
        num_elites = int(population_size * 0.1)
        elites = sorted_population[:num_elites]

        # Create the offspring
        offspring = []
        for i in range(population_size - num_elites):
            # Sample a parent from the population using tournament selection
            parent = tournament_selection(population, fitness, tournament_size)

            # Create an offspring by mutating the parent
            mutated_offspring = mutate(parent, mutation_rate, fpga_width, fpga_height, blocks)
            offspring.append(mutated_offspring)

        # Add the elites to the offspring
        offspring += elites

        # Replace the population with the offspring
        population = offspring

        # Evaluate the fitness of the offspring
        fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Return the best chromosome
    best_chromosome = population[fitness.index(min(fitness))]

    return best_chromosome


def evolutionary_strategy_bounding(fpga_width, fpga_height, blocks, population_size, mutation_rate, num_generations,
                                   tournament_size, max_time, desired_bounding_area):
    print('inside ESB')
    # Initialize the population
    population = [create_chromosome(fpga_width, fpga_height, blocks) for _ in range(population_size)]
    # Evaluate the fitness of each chromosome
    fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]
    bounding_areas = [bounding_area(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Set the start time
    start_time = time.time()

    # Run the ES until the desired bounding area is found or the time limit is reached
    while True:
        # Sort the population by their fitness values
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]
        sorted_fitness = sorted(fitness)
        sorted_bounding_areas = [x for _, x in sorted(zip(fitness, bounding_areas), key=lambda pair: pair[0])]

        # Check if any chromosome satisfies the desired bounding area
        for i in range(len(sorted_population)):
            if sorted_bounding_areas[i] <= desired_bounding_area:
                return sorted_population[i]

        # Keep the top 10% of the best individuals (elites)
        num_elites = int(population_size * 0.1)
        elites = sorted_population[:num_elites]

        # Create the offspring
        offspring = []
        for i in range(population_size - num_elites):
            # Sample a parent from the population using tournament selection
            parent = tournament_selection(population, fitness, tournament_size)
            # print(bounding_area(parent, fpga_width, fpga_height))
            # Create an offspring by mutating the parent
            mutated_offspring = mutate(parent, mutation_rate, fpga_width, fpga_height, blocks)
            offspring.append(mutated_offspring)

        # Add the elites to the offspring
        offspring += elites

        # Replace the population with the offspring
        population = offspring

        # Evaluate the fitness and bounding area of the offspring
        fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]
        bounding_areas = [bounding_area(chromosome, fpga_width, fpga_height) for chromosome in population]
        for i in range(len(bounding_areas)):
            if bounding_areas[i] <= desired_bounding_area:
                return population[i]

        # Check if the time limit has been reached
        if (time.time() - start_time) > max_time:
            break

    # If the desired bounding area was not found, return the chromosome with the closest bounding area
    closest_chromosome = sorted_population[sorted_bounding_areas.index(min(sorted_bounding_areas))]
    return closest_chromosome


def read_input_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
        # print(data)
    outline = data[0].strip().split()
    # print(outline)
    fpga_width, fpga_height = int(outline[1]), int(outline[2])
    block_sizes = {}
    for line in data[1:]:
        # print(line.strip().split())
        block, size = line.strip().split()
        block_sizes[block] = int(size)
    return fpga_width, fpga_height, block_sizes


def read_experiment_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    # outline = data[0].strip().split()
    fpga_width, fpga_height = 20, 20
    block_sizes = {}
    for line in data[0:]:
        block, size = line.strip().split()
        block_sizes[block] = int(size)
    return fpga_width, fpga_height, block_sizes


#get the file name and algorithm from the command line
# if len(sys.argv) != 3:
#     print("Usage: python your_script.py <file_name> <algorithm>")
#     exit()
#
# filename = sys.argv[1]
# algorithm = sys.argv[2]
#
# # Read input from the file
# fpga_width, fpga_height, block_sizes = read_input_file(filename)
#
# if algorithm == "ES":
#     total_blocks = len(block_sizes)
#     population_size = 4 * total_blocks
#     mutation_rate = 0.06  # 0.03 - 0.08 is good
#     num_generations = 150
#     tournament_size = 2 * total_blocks
#     start_time = time.time()
#     best_chromosome = evolutionary_strategy(fpga_width, fpga_height, block_sizes, population_size, mutation_rate,
#                                             num_generations,
#                                             tournament_size)
#     visualize(best_chromosome, fpga_width, fpga_height)
#     end_time = time.time()
#
#     runtime = end_time - start_time
#     print(runtime)
#     final_bounding_area = bounding_area(best_chromosome, fpga_width, fpga_height)
#     print(final_bounding_area)
#     print(len(block_sizes), len(best_chromosome))
#
# elif algorithm == "EST":
#     total_blocks = len(block_sizes)
#     population_size = 4 * total_blocks
#     mutation_rate = 0.06  # 0.03 - 0.08 is good
#     num_generations = 150
#     tournament_size = 2 * total_blocks
#     max_time = 240
#     best_chromosome_timed = evolutionary_strategy_timed(fpga_width, fpga_height, block_sizes, population_size,
#                                                         mutation_rate, num_generations,
#                                                         tournament_size, max_time)
#     final_bounding_area = bounding_area(best_chromosome_timed, fpga_width, fpga_height)
#     print(final_bounding_area)
#     for gene in best_chromosome_timed:
#         print(gene)
#     print(len(block_sizes), len(best_chromosome_timed))
#     visualize(best_chromosome_timed, fpga_width, fpga_height)
#
# elif algorithm == "ESB":
#     total_blocks = len(block_sizes)
#     population_size = 5 * total_blocks
#     mutation_rate = 0.06  # 0.03 - 0.08 is good
#     num_generations = 150
#     tournament_size = 2 * total_blocks
#     max_time = 300
#     desired_bounding_area = 98
#     start_time = time.time()
#     best_chromosome_bounding = evolutionary_strategy_bounding(fpga_width, fpga_height, block_sizes, population_size, mutation_rate, num_generations,
#                                    tournament_size, max_time, desired_bounding_area)
#     end_time = time.time()
#
#     runtime = end_time - start_time
#     print(runtime)
#     final_bounding_area = bounding_area(best_chromosome_bounding, fpga_width, fpga_height)
#     print(final_bounding_area)
#     for gene in best_chromosome_bounding:
#         print(gene)
#     print(len(block_sizes), len(best_chromosome_bounding))
#     visualize(best_chromosome_bounding, fpga_width, fpga_height)
# else:
#     print('Invalid algorithm')

df = pd.read_csv('optimal_LBMA_final.csv', delimiter=';', usecols=['filename', 'final_bounding_area', 'runtime'])
#this code for testing purposes
folder_path = "../Test data/LBMA_Data"
files = os.listdir(folder_path)
#files = ['MBLA_15.txt','MBLA_21.txt','MBLA_22.txt']
results = []
max_attempts = 3  # maximum number of attempts to reach the required bounding area
#filtered_files = [filename for filename in files if filename.endswith(".txt") and filename in df['filename'].values]
#files = ['MBLA_17.txt','MBLA_18.txt','MBLA_28.txt','MBLA_46.txt']
#exceptional_files = ['MBLA_0.txt','MBLA_1.txt','MBLA_10.txt','MBLA_11.txt','MBLA_12.txt','MBLA_13.txt','MBLA_14.txt','MBLA_15.txt','MBLA_16.txt','MBLA_17.txt']
for filename in files:
    if filename.endswith(".txt"):
        print(filename)
        filepath = os.path.join(folder_path, filename)
        fpga_width, fpga_height, block_sizes = read_input_file(filepath)
        total_blocks = len(block_sizes)
        population_size = 5 * total_blocks
        mutation_rate = 0.06 # 0.03 - 0.08 is good
        num_generations = 150
        tournament_size = 3 * total_blocks
        max_time = 420
        desired_bounding_area = df.loc[df['filename'] == filename, 'final_bounding_area'].iloc[0]
        min_bounding_area = float('inf')
        min_attempt_details = None
        attempt = 1
        while attempt <= max_attempts:
            start_time = time.time()
            best_chromosome_bounded = evolutionary_strategy_bounding(fpga_width, fpga_height, block_sizes,population_size, mutation_rate, num_generations,tournament_size,max_time, desired_bounding_area)
            end_time = time.time()

            runtime = end_time - start_time
            final_bounding_area = bounding_area(best_chromosome_bounded, fpga_width, fpga_height)
            print(f"Attempt {attempt}, Final bounding area: {final_bounding_area}")

            if final_bounding_area <= desired_bounding_area:
                # Store the results for this file
                results.append({
                    'filename': filename,
                    'final_bounding_area': final_bounding_area,
                    'runtime': runtime,
                    'total_blocks': len(block_sizes),
                    'final_num_blocks': len(best_chromosome_bounded),
                    'num_attempts': attempt
                })
                break  # Exit the while loop if the required bounding area is met
            # Check if this attempt has the minimum bounding area so far
            if final_bounding_area < min_bounding_area:
                min_bounding_area = final_bounding_area
                min_attempt_details = {
                    'filename': filename,
                    'final_bounding_area': final_bounding_area,
                    'runtime': runtime,
                    'total_blocks': len(block_sizes),
                    'final_num_blocks': len(best_chromosome_bounded),
                    'num_attempts': attempt
                }

            attempt += 1

            # If the desired bounding area was not met, write details of attempt with min bounding area
        if attempt > max_attempts and min_attempt_details is not None:
            results.append(min_attempt_details)

        # Write the results to a CSV file
        with open('LBMA_reach_ES_optimal', 'w', newline='') as csvfile:
            fieldnames = ['filename', 'final_bounding_area', 'runtime', 'total_blocks', 'final_num_blocks', 'num_attempts']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)

# folder_path = "../Test data/LBMA_Data"
# files = os.listdir(folder_path)
# results = []
# #max_attempts = 5  # maximum number of attempts to reach the required bounding area
# #files = ['MBLA_17.txt','MBLA_18.txt','MBLA_28.txt','MBLA_46.txt']
# for filename in files:
#     if filename.endswith(".txt"):
#         print(filename)
#         filepath = os.path.join(folder_path, filename)
#         fpga_width, fpga_height, block_sizes = read_input_file(filepath)
#         total_blocks = len(block_sizes)
#         population_size = 4 * total_blocks
#         mutation_rate = 0.06  # 0.03 - 0.08 is good
#         num_generations = 150
#         tournament_size = 2 * total_blocks
#         max_time = 360
#         #req_bounding_area = df.loc[df['filename'] == filename, 'final_bounding_area'].iloc[0]
#
#
#         start_time = time.time()
#         best_chromosome_timed = evolutionary_strategy_timed(fpga_width, fpga_height, block_sizes, population_size, mutation_rate, num_generations,tournament_size, max_time)
#
#         end_time = time.time()
#
#         runtime = end_time - start_time
#         final_bounding_area = bounding_area(best_chromosome_timed, fpga_width, fpga_height)
#         results.append({
#                 'filename': filename,
#                 'final_bounding_area': final_bounding_area,
#                 'runtime': runtime,
#                 'total_blocks': len(block_sizes),
#                 'final_num_blocks': len(best_chromosome_timed)
#             })
#         print(f"Filename {filename}, Final bounding area: {final_bounding_area}")
#
#         # if final_bounding_area <= req_bounding_area:
#         #     # Store the results for this file
#         #     results.append({
#         #         'filename': filename,
#         #         'final_bounding_area': final_bounding_area,
#         #         'runtime': runtime,
#         #         'total_blocks': len(block_sizes),
#         #         'final_num_blocks': len(final_layout),
#         #         'num_attempts': attempt
#         #     })
#         #         break  # Exit the while loop if the required bounding area is met
#         #
#         #     attempt += 1
#
#         # Write the results to a CSV file
#         with open('final_LBMA_timed_ES_6min.csv', 'w', newline='') as csvfile:
#             fieldnames = ['filename', 'final_bounding_area', 'runtime', 'total_blocks', 'final_num_blocks']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#             writer.writeheader()
#             for result in results:
#                 writer.writerow(result)


# folder_path = "../Test data/Final_experiment_data"
# files = os.listdir(folder_path)
# results = []
# max_attempts = 3  # maximum number of attempts to reach the required bounding area
#
# for filename in files:
#     if filename.endswith(".txt"):
#         print(filename)
#         filepath = os.path.join(folder_path, filename)
#         fpga_width, fpga_height, block_sizes = read_experiment_file(filepath)
#         total_blocks = len(block_sizes)
#         population_size = 4 * total_blocks
#         mutation_rate = 0.06  # 0.03 - 0.08 is good
#         num_generations = 150
#         tournament_size = 2 * total_blocks
#         max_time = 180
#
#         min_bounding_area = float('inf')
#         min_result = None
#
#         for attempt in range(max_attempts):
#             start_time = time.time()
#             best_chromosome_timed = evolutionary_strategy(fpga_width, fpga_height, block_sizes, population_size, mutation_rate,num_generations,
#                                              tournament_size)
#             end_time = time.time()
#
#             runtime = end_time - start_time
#             final_bounding_area = bounding_area(best_chromosome_timed, fpga_width, fpga_height)
#
#             print(f"Filename {filename}, Attempt {attempt + 1}, Final bounding area: {final_bounding_area}")
#
#             if final_bounding_area < min_bounding_area:
#                 min_bounding_area = final_bounding_area
#                 min_result = {
#                     'filename': filename,
#                     'final_bounding_area': final_bounding_area,
#                     'runtime': runtime,
#                     'total_blocks': len(block_sizes),
#                     'final_num_blocks': len(best_chromosome_timed),
#                     'num_attempts': attempt + 1
#                 }
#
#         # Write the results to a CSV file
#         with open('final_experiment_file_ES_normal.csv', 'a', newline='') as csvfile:
#             fieldnames = ['filename', 'final_bounding_area', 'runtime', 'total_blocks', 'final_num_blocks',
#                           'num_attempts']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#             if os.stat('final_experiment_file.csv').st_size == 0:
#                 writer.writeheader()
#
#             writer.writerow(min_result)
