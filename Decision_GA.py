import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sympy
import os
import csv
import time
import sys


# for creating intial chromsome
def create_chromosome(fpga_width, fpga_height, block_sizes, preplaced_blocks):
    chromosome = []
    placed_blocks = set()
    block_shapes = {}
    available_micro_slots = fpga_width * fpga_height - sum(
        [preplaced_block["width"] * preplaced_block["height"] for preplaced_block in preplaced_blocks])
    required_area = sum([block_size for block_size in block_sizes.values()])
    if required_area > available_micro_slots:
        print("Need a bigger FPGA because area requirement is more than available micro slots")
        return None
    # Calculate block shapes
    for block, area in block_sizes.items():
        if sympy.isprime(area) and (area > fpga_height): area = area + 1
        width, height = get_shape(area)
        block_shapes[block] = (width, height)
    # Place the preplaced block
    for preplaced_block in preplaced_blocks:
        x = preplaced_block["x"]
        y = preplaced_block["y"]
        width = preplaced_block["width"]
        height = preplaced_block["height"]
        name = preplaced_block["name"]
        placed_blocks.add(name)
    sorted_blocks = sorted(block_sizes.items(), key=lambda x: x[1], reverse=True)
    for block, area in sorted_blocks:
        width, height = block_shapes[block]
        placed = False
        count = 0
        while not placed and count < 1000:
            count += 1
            x = random.randint(0, fpga_width)
            y = random.randint(0, fpga_height)
            if check_validity(fpga_width, fpga_height, x, y, width, height, preplaced_blocks):
                chromosome.append((block, x, y, width, height))
                placed = True
            if count >= 1000:
                print(f"Block {block} couldn't be placed.")
    return chromosome


# get the shape (width,height) for the given area
def get_shape(area):
    width = int(math.sqrt(area))
    height = int(area / width)
    while width * height != area:
        width -= 1
        height = int(area / width)
    return width, height


# get all the possible rectangle shapes with the given area
def get_rectangles(area):
    widths = []
    heights = []
    for i in range(1, int(math.sqrt(area)) + 1):
        if area % i == 0:
            widths.append(i)
            heights.append(area // i)
    rectangles = []
    for w, h in zip(widths, heights):
        rectangles.append((w, h))
        if w != h:
            rectangles.append((h, w))
    return rectangles


# check function to see that the blocks are not placed outside FPGA and on preplaced blocks
def check_validity(fpga_width, fpga_height, x, y, width, height, preplaced_blocks):
    # Check if block is outside the boundaries

    if x + width > fpga_width or y + height > fpga_height:
        return False

    # Check if block overlaps with preplaced blocks
    for preplaced_block in preplaced_blocks:
        px, py = preplaced_block['x'], preplaced_block['y']
        p_x, p_y = px + preplaced_block['width'], py + preplaced_block['height']

        # Check if new block is completely inside the preplaced block
        if px <= x < p_x and py <= y < p_y and x + width <= p_x and y + height <= p_y:
            return False

        # Check if new block completely contains the preplaced block
        if x <= px < x + width and y <= py < y + height and x + width >= p_x and y + height >= p_y:
            return False

        # Check if new block overlaps with preplaced block
        if x < p_x and x + width > px and y < p_y and y + height > py:
            return False

    return True


# visualize the chromosome
def visualize(chromosome, fpga_width, fpga_height, preplaced_blocks):
    plt.figure(figsize=(10, 10))
    plt.xlim(0, fpga_width)
    plt.ylim(0, fpga_height)
    areas = [block[3] * block[4] for block in chromosome]
    normalized_areas = [(area - min(areas)) / (max(areas) - min(areas)) for area in areas]
    # Create a colormap
    cmap = cm.get_cmap("viridis")

    for preplaced_block in preplaced_blocks:
        if preplaced_block["name"].startswith("preplaced"):
            color = "black"
            plt.gca().add_patch(plt.Rectangle((preplaced_block["x"], preplaced_block["y"]), preplaced_block["width"],
                                              preplaced_block["height"],
                                              facecolor=color, fill=True, edgecolor='black'))
    # plt.gca().set_aspect('equal', adjustable='box')
    for i, block in enumerate(chromosome):
        color = cmap(normalized_areas[i])
        plt.gca().add_patch(plt.Rectangle((block[1], block[2]), block[3], block[4], color=color, fill=True))
        plt.gca().text(block[1] + block[3] / 2, block[2] + block[4] / 2, block[0], ha='center', va='center')
        # plt.gca().text(block[1] + math.sqrt(block[3])/2, block[2] + math.sqrt(block[4])/2, areas[i], ha='left', va='bottom')
    # Add x and y axis labels
    # plt.xticks(range(fpga_width+1))
    # plt.yticks(range(fpga_height+1))
    plt.gca().set_xticks(range(fpga_width + 1))
    plt.gca().set_yticks(range(fpga_height + 1))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().grid(True, which='both', color='grey', linewidth=1.0)
    plt.show()


# count the number overlaps between the blocks
def count_overlaps(chromosome):
    count = 0
    for i, (name1, x1, y1, w1, h1) in enumerate(chromosome):
        rect1 = (x1, y1, x1 + w1, y1 + h1)
        for j in range(i + 1, len(chromosome)):
            name2, x2, y2, w2, h2 = chromosome[j]
            rect2 = (x2, y2, x2 + w2, y2 + h2)
            if do_rectangles_overlap(rect1, rect2):
                count += 1
    return count


# checking overlaps between two rectangles
def do_rectangles_overlap(rect1, rect2):
    return rect1[0] < rect2[2] and rect1[2] > rect2[0] and rect1[1] < rect2[3] and rect1[3] > rect2[1]


# finding new x and y for the given block
max_retries = 1000


def find_new_positions(block, fpga_width, fpga_height):
    x, y = block[1], block[2]
    width, height = block[3], block[4]
    changed = False
    for _ in range(max_retries):
        new_x = random.randint(0, fpga_width - 1)
        new_y = random.randint(0, fpga_height - 1)
        if check_validity(fpga_width, fpga_height, new_x, new_y, width, height, preplaced_blocks):
            return new_x, new_y
    # If the new block is not valid after max_retries, return the original block position
    return x, y

# get new shape for the given block in chromosome
def get_new_shape(block, block_area, fpga_width, fpga_height):
    shape_of_block = (block[3], block[4])
    x, y = block[1], block[2]
    possible_shapes = get_rectangles(block_area)
    for w, h in possible_shapes:
        if (w, h) != shape_of_block:
            new_x, new_y = find_new_positions((block[0], x, y, w, h), fpga_width, fpga_height)
            if new_x != x and new_y != y:
                new_block = (block[0], new_x, new_y, w, h)
                return new_block
        else:
            new_x, new_y = find_new_positions((block[0], block[1], block[2], w, h), fpga_width, fpga_height)
            if new_x != block[1] or new_y != block[2]:
                # Update the block with the new position
                new_block = (block[0], new_x, new_y, block[3], block[4])
                return new_block
    return block

#mutate function to get new chromosome by changing the shape or changing the position
def mutate(chromosomes, mutation_rate, fpga_width, fpga_height, blocks_dict):
    for i in range(len(chromosomes)):
        if random.uniform(0, 1) < mutation_rate:
            block_name, x, y, width, height = chromosomes[i]
            block_raw_area = blocks_dict[block_name]
            if random.uniform(0, 1) < 0.5:
                #print('in changing shape')
                updated_block = get_new_shape((block_name, x, y, width, height),block_raw_area,fpga_width,fpga_height)
                chromosomes[i] = (block_name,updated_block[1],updated_block[2],updated_block[3],updated_block[4])
            else:
                #print('in changing position')
                new_x,new_y = find_new_positions((block_name, x, y, width, height),fpga_width,fpga_height)
                chromosomes[i] = (block_name, new_x, new_y, width, height)

    return chromosomes

# tounament selection function
def tournament_selection(population, fitness, tournament_size):
    # Select a random subset of the population
    subset = random.sample(list(range(len(population))), tournament_size)
    # Choose the individual with the best fitness in the subset
    winner = subset[0]
    for i in subset:
        if fitness[i] < fitness[winner]:
            winner = i
    return population[winner]



def evolutionary_strategy(fpga_width, fpga_height, blocks, preplaced_blocks, population_size, mutation_rate,
                          num_generations, tournament_size):
    # Initialize the population
    population = [create_chromosome(fpga_width, fpga_height, blocks, preplaced_blocks) for _ in range(population_size)]
    # Evaluate the fitness of each chromosome
    fitness = [count_overlaps(chromosome) for chromosome in population]
    #print(fitness)
    for i in fitness:
        if i == 0:
            best_chromosome = population[fitness.index(i)]
            return best_chromosome

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

            # Create an offspring by mutating the parent
            mutated_offspring = mutate(parent, mutation_rate, fpga_width, fpga_height, blocks)
            offspring.append(mutated_offspring)

        # Add the elites to the offspring
        offspring += elites

        # Replace the population with the offspring
        population = offspring

        # Evaluate the fitness of the offspring
        fitness = [count_overlaps(chromosome) for chromosome in population]
        for i in fitness:
            if i == 0:
                best_chromosome = population[fitness.index(i)]
                return best_chromosome

    # Return the best chromosome
    best_chromosome = population[fitness.index(min(fitness))]

    return best_chromosome

# read the given file and return fpga_width, fpga_height, block_sizes
def read_input_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
        #print(data)
    outline = data[0].strip().split()
    #print(outline)
    fpga_width, fpga_height = int(outline[1]), int(outline[2])
    block_sizes = {}
    for line in data[1:]:
        #print(line.strip().split())
        block, size = line.strip().split()
        block_sizes[block] = int(size)
    return fpga_width, fpga_height, block_sizes

def final_accept(chromosome):
    if count_overlaps(chromosome) == 0:
        return True
    else:
        return False
#
# filename = "..\Test Data\Final_decision_data\Zync7200_4.txt"
# fpga_width, fpga_height, block_sizes = read_input_file(filename)
# preplaced_blocks = [{"name": "preplaced_block1", "x": 0, "y": 1, "width": 2, "height": 2},
#                     {"name": "preplaced_block2", "x": 3, "y": 0, "width": 1, "height": 3}]
# total_blocks = len(block_sizes)
# population_size = 4 * total_blocks
# mutation_rate = 0.08  # 0.03 - 0.08 is good
# num_generations = 100
# tournament_size = 2 * total_blocks
# best_chromosome = evolutionary_strategy(fpga_width, fpga_height, block_sizes, preplaced_blocks, population_size, mutation_rate, num_generations,tournament_size)
# final_acceptance = final_accept(best_chromosome)
# print(final_acceptance)
# visualize(best_chromosome,fpga_width,fpga_height, preplaced_blocks)

#get the file name and algorithm from the command line
# if len(sys.argv) != 3:
#     print("Usage: python your_script.py <file_name> <device>")
#     exit()
#
# filename = sys.argv[1]
# device = sys.argv[2]
# #Read input from the file
# fpga_width, fpga_height, block_sizes = read_input_file(filename)
# if device == "Zynq7200":
#     preplaced_blocks = [{"name": "preplaced_block1", "x": 0, "y": 1, "width": 2, "height": 2},
#                         {"name": "preplaced_block2", "x": 3, "y": 0, "width": 1, "height": 3}]
#
#     best_chromosome = evolutionary_strategy(fpga_width, fpga_height, block_sizes, preplaced_blocks, population_size, mutation_rate, num_generations,tournament_size)
#     final_acceptance = final_accept(best_chromosome)
#     print(final_acceptance)
#     visualize(best_chromosome,fpga_width,fpga_height, preplaced_blocks)
#
# if device == "Zynq7030":
#     preplaced_blocks = [{"name": "preplaced_block1", "x": 0, "y": 2, "width": 2, "height": 2},
#                         {"name": "preplaced_block2", "x": 4, "y": 0, "width": 1, "height": 4},
#                         {"name": "preplaced_block3", "x": 6, "y": 0, "width": 1, "height": 1}]
#     best_chromosome = evolutionary_strategy(fpga_width, fpga_height, block_sizes, preplaced_blocks, population_size, mutation_rate, num_generations,tournament_size)
#     final_acceptance = final_accept(best_chromosome)
#     print(final_acceptance)
#     visualize(best_chromosome,fpga_width,fpga_height, preplaced_blocks)

folder_path = "../Test data/Final_decision_data"
files = os.listdir(folder_path)
results = []

for filename in files:
    if filename.endswith(".txt"):
        print(filename)
        filepath = os.path.join(folder_path, filename)
        fpga_width, fpga_height, blocks = read_input_file(filepath)
        total_blocks = len(blocks)
        population_size = 4 * total_blocks
        # print(population_size)
        preplaced_blocks = [{"name": "preplaced_block1", "x": 0, "y": 1, "width": 2, "height": 2},
                            {"name": "preplaced_block2", "x": 3, "y": 0, "width": 1, "height": 3}]
        mutation_rate = 0.06  # 0.03 - 0.08 is good
        num_generations = 100
        tournament_size = 2 * total_blocks

        max_iterations = 10
        iterations = 0
        final_acceptance = False
        while not final_acceptance and iterations < max_iterations:
            start_time = time.time()
            final_chromosome = evolutionary_strategy(fpga_width, fpga_height, blocks,preplaced_blocks, population_size, mutation_rate, num_generations,tournament_size)
            end_time = time.time()

            runtime = end_time - start_time
            final_acceptance = final_accept(final_chromosome)
            print(final_acceptance)

            iterations += 1

        # Store the results for this file
        results.append({
            'filename': filename,
            'final_acceptance': final_acceptance,
            'runtime': runtime
        })

# Write the results to a CSV file
with open('Final_Decision_GA_results_new.csv', 'w', newline='') as csvfile:
    fieldnames = ['filename', 'final_acceptance', 'runtime']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(result)