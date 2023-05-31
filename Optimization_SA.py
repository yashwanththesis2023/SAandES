import os
import time
import random
import math
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import pandas as pd


# this function for creating the initial layout choosing x and y points for each slot randomly.
#input: fpga_width, fpga_height, block_sizes
#output: layout
def generate_layout(fpga_width, fpga_height):
    layout = []
    placed_blocks = set()
    sorted_blocks = sorted(block_sizes.items(), key=lambda x: x[1], reverse=True)
    max_attempts = 1500# set a maximum number of attempts to place a block
    for block, area in sorted_blocks:
        if block in placed_blocks:
            continue
        placed = False
        attempts = 0
        #placing the blocks
        while not placed and attempts < max_attempts:
            x = random.randint(0, fpga_width)
            y = random.randint(0, fpga_height)
            width, height = area_to_rectangle(area)
            rectangle = Rectangle((x, y), width, height, color=np.random.rand(3,))
            rectangle.set_label(block)
            if check_validity(layout + [rectangle], fpga_width, fpga_height):
                layout.append(rectangle)
                placed_blocks.add(block)
                placed = True
            attempts += 1
        if attempts == max_attempts and not placed:
          print(f"Block {block} could not be placed within {max_attempts} attempts.")
    #print("initial layout created")
    return layout


# this function provides the width and height of the block when you give the input as required area.
def area_to_rectangle(area):
    width = int(math.sqrt(area))
    height = int(area / width)
    while width * height != area:
        width -= 1
        height = int(area / width)
    return width, height

def new_area_to_rectangle(area):
    if area > 1 and all(area % i != 0 for i in range(2, int(math.sqrt(area))+1)):
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

# this function checks whether the blocks in the layout has overlaps or outside the boundaries of given FPGA
def check_validity(layout, fpga_width, fpga_height):
    # check for overlapping and boundaries
    coordinates = set()
    for rectangle in layout:
        x0, y0 = rectangle.get_xy()
        x1 = x0 + rectangle.get_width()
        y1 = y0 + rectangle.get_height()

        # check if block is outside the boundaries
        if x1 > fpga_width or y1 > fpga_height:
            return False

        # check for overlapping
        for x in range(int(x0), int(x1)):
            for y in range(int(y0), int(y1)):
                if (x, y) in coordinates:
                    return False
                coordinates.add((x, y))
    coordinates.clear()  # clear the coordinates set after all blocks have been checked
    return True


# this function is used to visualize the layout.
def visualize(layout):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Compute the area of each rectangle
    areas = [rectangle.get_width() * rectangle.get_height() for rectangle in layout]

    # Normalize the areas to be between 0 and 1
    normalized_areas = [(area - min(areas)) / (max(areas) - min(areas)) for area in areas]

    # Create a colormap
    cmap = cm.get_cmap("viridis")

    for i, rectangle in enumerate(layout):
        color = cmap(normalized_areas[i])
        rect = plt.Rectangle((rectangle.get_xy()), rectangle.get_width(), rectangle.get_height(), color=color,
                             fill=True, edgecolor='black')
        ax.add_patch(rect)

        # Find the corresponding name of the rectangle
        # Add the block name and area to the center of the patch
        ax.annotate("{}\n{:.2f}".format(rectangle.get_label(), areas[i]), (
            rectangle.get_xy()[0] + rectangle.get_width() / 2, rectangle.get_xy()[1] + rectangle.get_height() / 2),
                    color='black', fontsize=18, ha='center', va='center')

    plt.xlim(0, fpga_width)
    plt.ylim(0, fpga_height)
    plt.xlabel('fpga_width', fontsize=20)
    plt.ylabel('fpga_height', fontsize=20)
    #plt.savefig('final_opti_SA_sample')
    plt.show()


# this function returns the cost of the given layout. It also penalizes the unused area in the bounding rectangle area.
def cost(layout):
    # Compute the area of the bounding rectangle
    x_coords = [rect.get_xy()[0] for rect in layout]
    y_coords = [rect.get_xy()[1] for rect in layout]
    x0 = min(x_coords)
    y0 = min(y_coords)
    x1 = max([x + rect.get_width() for x, rect in zip(x_coords, layout)])
    y1 = max([y + rect.get_height() for y, rect in zip(y_coords, layout)])
    bounding_rect_area = (x1 - x0) * (y1 - y0)
    unused_area = bounding_rect_area - sum([rect.get_width() * rect.get_height() for rect in layout])
    cost = bounding_rect_area + 0.1 * unused_area  # add a penalty of 0.1 times the unused area
    return cost

#this function provides the bounding area of the layout
def bounding_area(layout):
    x_coords = [rect.get_xy()[0] for rect in layout]
    y_coords = [rect.get_xy()[1] for rect in layout]
    x0 = min(x_coords)
    y0 = min(y_coords)
    x1 = max([x + rect.get_width() for x, rect in zip(x_coords, layout)])
    y1 = max([y + rect.get_height() for y, rect in zip(y_coords, layout)])
    # print('x0 =', x0)
    # print('y0 =', y0)
    # print('x1 =', x1)
    # print('y1 =', y1)
    MBR_area = (x1 - x0) * (y1 - y0)
    return MBR_area

def final_bounding_area(layout):
    x_coords = [rect.get_xy()[0] for rect in layout]
    y_coords = [rect.get_xy()[1] for rect in layout]
    x0 = min(x_coords)
    y0 = min(y_coords)
    x1 = max([x + rect.get_width() for x, rect in zip(x_coords, layout)])
    y1 = max([y + rect.get_height() for y, rect in zip(y_coords, layout)])
    # print('x0 =', x0)
    # print('y0 =', y0)
    # print('x1 =', x1)
    # print('y1 =', y1)
    width = (x1-x0)
    MBR_AREA = (x1 - x0) * (y1 - y0)
    return MBR_AREA, width

#this function tries to find new coordinates for the block for max_retries.

max_retries=1000
def find_new_positions(block, fpga_width, fpga_height, current_layout):
    changed = 0
    x,y = block.get_xy()
    new_width = block.get_width()
    new_height = block.get_height()

    # Get the bounding coordinates of the current layout
    x_coords = [rect.get_xy()[0] for rect in current_layout]
    y_coords = [rect.get_xy()[1] for rect in current_layout]
    x0 = min(x_coords)
    y0 = min(y_coords)
    x1 = max([x + rect.get_width() for x, rect in zip(x_coords, current_layout)])
    y1 = max([y + rect.get_height() for y, rect in zip(y_coords, current_layout)])

    for i in range(max_retries):
        # Generate a new position for the block randomly within the bounding area
        new_x = int(random.uniform(x0, x1 - new_width))
        new_y = int(random.uniform(y0, y1 - new_height))
        if new_x != x or new_y != y:
          block.set_xy((new_x, new_y))

        # Check if the new block is still inside the FPGA boundaries and does not overlap with other blocks
          if check_validity(current_layout, fpga_width, fpga_height):
              changed = 1
              return changed

    # If the new block is not valid after max_retries, return the original block
    block.set_xy((x, y))
    changed = 0
    return changed
def calculate_aspect_ratio_bounds(block_area):
    divisors = []
    for i in range(1, int(math.sqrt(block_area)) + 1):
        if block_area % i == 0:
            divisors.append(i)

    min_ratio = float("inf")
    max_ratio = 0.0
    for i in range(len(divisors)):
        width = divisors[i]
        height = block_area // width
        aspect_ratio = height / width
        if aspect_ratio < min_ratio:
            min_ratio = aspect_ratio
        if aspect_ratio > max_ratio:
            max_ratio = aspect_ratio

    return min_ratio, max_ratio
#this function change the current shape of the block to new shape which can reduce the total bounding area.
def select_best_shape(block, block_area, current_layout, fpga_width, fpga_height):
    widths = []
    heights = []
    for i in range(1, int(math.sqrt(block_area)) + 1):
        if block_area % i == 0:
            widths.append(i)
            heights.append(block_area // i)
    shapes = []
    min_aspect_ratio, max_aspect_ratio = calculate_aspect_ratio_bounds(block_area)
    for width, height in zip(widths, heights):
        block.set_width(width)
        block.set_height(height)
        #print('-------',block.get_width(), block.get_height())
        if check_validity(current_layout, fpga_width, fpga_height) and min_aspect_ratio <= height / width <= max_aspect_ratio and width * height == block_area:
            shapes.append((width, height))
    if shapes:
        # choose the shape with the best potential for reducing the total bounding area
        shapes = sorted(shapes, key=lambda shape: shape[0])
        best_shape = shapes[0]
        block.set_width(best_shape[0])
        block.set_height(best_shape[1])
    else:
        # if there are no valid shapes within the given aspect ratio bounds or with the same area, return the original shape
        pass
    #print(block.get_width(), block.get_height())

    return block


#this function change the current shape of the block to new shape which can reduce the total bounding area.
# def select_best_shape(block, block_area, current_layout, fpga_width, fpga_height):
#     # Calculate the current bounding area
#     current_bounding_area = bounding_area(current_layout)
#
#     # Calculate the aspect ratio of the block
#     aspect_ratio = block.get_height() / block.get_width()
#
#     # Define the range of valid widths and heights
#     max_width = min(fpga_width, math.floor(math.sqrt(block_area)))
#     max_height = min(fpga_height, math.floor(math.sqrt(block_area)))
#     min_width = max(1, math.ceil(block_area / fpga_height))
#     min_height = max(1, math.ceil(block_area / fpga_width))
#
#     # Initialize the best shape and the change in bounding area
#     best_shape = (block.get_width(), block.get_height())
#     best_delta_bounding_area = 0
#
#     # Randomly sample new widths and heights
#     for _ in range(10):
#         new_width = random.randint(min_width, max_width)
#         new_height = random.randint(min_height, max_height)
#         if new_width * new_height == block_area:
#             # Calculate the change in aspect ratio
#             delta_aspect_ratio = abs(new_height / new_width - aspect_ratio)
#
#             # Calculate the change in bounding area
#             block.set_width(new_width)
#             block.set_height(new_height)
#             if check_validity(current_layout, fpga_width, fpga_height):
#                 new_bounding_area = bounding_area(current_layout)
#                 delta_bounding_area = new_bounding_area - current_bounding_area
#             else:
#                 delta_bounding_area = float('inf')
#
#             # Accept or reject the change based on the Metropolis criterion
#             if delta_bounding_area < best_delta_bounding_area:
#                 best_shape = (new_width, new_height)
#                 best_delta_bounding_area = delta_bounding_area
#
#     # Set the block's width and height to the best shape
#     block.set_width(best_shape[0])
#     block.set_height(best_shape[1])
#     return block

# this function will create a new layout by changing the shape of the block and then the position or just change the position of the selected block.
# def generate_new_layout(current_layout,fpga_width,fpga_height):
#   new_layout = current_layout.copy()
#   block = random.choice(new_layout)
#   block_area = block.get_width() * block.get_height()
#   if random.uniform(0, 1) < 0.5:
#     new_shaped_block = select_best_shape(block,block_area,current_layout,fpga_width, fpga_height)
#     changed = find_new_positions(new_shaped_block, fpga_width, fpga_height, current_layout)
#   else:
#     changed = find_new_positions(block, fpga_width, fpga_height, current_layout)
#   return new_layout, changed
def rotate_block(block,fpga_width, fpga_height,current_layout):
    x, y = block.get_xy()
    current_width = block.get_width()
    current_height = block.get_height()

    block.set_width(current_height)
    block.set_height(current_width)
    changed = find_new_positions(block, fpga_width, fpga_height, current_layout)
    if changed:
        changed = 1
        return changed
    else:
        block.set_width(current_width)
        block.set_height(current_height)
        changed = 0
        return changed



def generate_new_layout(current_layout, fpga_width, fpga_height, num_blocks=4):
    new_layout = current_layout.copy()
    changed = []
    for i in range(num_blocks):
        block = random.choice(new_layout)
        block_area = block.get_width() * block.get_height()
        if random.uniform(0, 1) < 0.5:
            new_shaped_block = select_best_shape(block, block_area, current_layout, fpga_width, fpga_height)
            changed = find_new_positions(new_shaped_block, fpga_width, fpga_height, current_layout)
        else:
            changed = find_new_positions(block, fpga_width, fpga_height, current_layout)

    return new_layout, changed
#
# import math
#
# def is_prime(n):
#     if n <= 1:
#         return False
#     elif n <= 3:
#         return True
#     elif n % 2 == 0 or n % 3 == 0:
#         return False
#     i = 5
#     while i * i <= n:
#         if n % i == 0 or n % (i + 2) == 0:
#             return False
#         i += 6
#     return True
#
# def factorize(n):
#     factors = []
#     d = 2
#     while d * d <= n:
#         while (n % d) == 0:
#             factors.append(d)
#             n //= d
#         d += 1
#     if n > 1:
#         factors.append(n)
#     return factors
#
# def read_input_file(filename):
#     with open(filename, 'r') as file:
#         data = file.readlines()
#     outline = data[0].strip().split()
#     fpga_width, fpga_height = int(outline[1]), int(outline[2])
#     block_sizes = {}
#     for line in data[1:]:
#         block, size = line.strip().split()
#         size = int(size)
#         if is_prime(size):
#             size += 1
#         else:
#             factors = factorize(size)
#             #and is_prime(factors[0]) and is_prime(factors[1])
#             if len(factors) == 2:
#                 size += 1
#         block_sizes[block] = size
#     return fpga_width, fpga_height, block_sizes

#this function reads the input file and returns fpga_width, fpga_height, block_sizes
def read_input_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    outline = data[0].strip().split()
    #
    fpga_width, fpga_height = int(outline[1]), int(outline[2])
    block_sizes = {}
    for line in data[1:]:
        block, size = line.strip().split()
        block_sizes[block] = int(size)
    return fpga_width, fpga_height, block_sizes
def read_experiment_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    # outline = data[0].strip().split()
    fpga_width, fpga_height = 20,20
    block_sizes = {}
    for line in data[0:]:
        block, size = line.strip().split()
        block_sizes[block] = int(size)
    return fpga_width, fpga_height, block_sizes


def initialize_temperature(current_layout, fpga_width, fpga_height, inner_loop):
    cost_diffs = []
    current_cost = cost(current_layout)
    for i in range(inner_loop):
        new_layout, _ = generate_new_layout(current_layout, fpga_width, fpga_height)
        if check_validity(new_layout, fpga_width, fpga_height):
            new_cost = cost(new_layout)
            cost_diff = new_cost - current_cost
            if cost_diff > 0:
                cost_diffs.append(cost_diff)
    if len(cost_diffs) > 0:
        cost_change_avg = sum(cost_diffs) / len(cost_diffs)
        P = 0.95  # desired acceptance rate
        T0 = -cost_change_avg / math.log(P)
    else:
        # if all perturbations result in downhill cost changes,
        # set T0 to a high value
        T0 = 10000000
    return T0


#required parameters for running adaptive Simulated Annealing loop
T = 10000000
T_init = 1000000
T_min = 0.001
alpha_norm = 0.6
alpha_min = 0.3
alpha_max = 0.8
alpha_factor = 0.1
inner_loop = 100
max_unchanged_iterations = 100
target_acceptance = 0.5
acceptance_window = 0.1

#this is the Simulated Annealing function where the annealing schedule is adaptive.
def simulated_annealing(fpga_width, fpga_height, T, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, max_unchanged_iterations, target_acceptance, acceptance_window):
    print('Im inside SA')
    current_layout = generate_layout(fpga_width, fpga_height)
    current_cost = cost(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    unchanged_iterations = 0
    temp = [T]
    while T > T_min and unchanged_iterations < max_unchanged_iterations:
        ##print(bounding_area(current_layout))
        unchanged_iterations = 0
        accepted_count = 0
        total_count = 0
        num_iterations = 0
        alpha = alpha_norm
        for i in range(inner_loop):
            #print(i)
            new_layout, changed = generate_new_layout(current_layout, fpga_width, fpga_height)
            if check_validity(new_layout, fpga_width, fpga_height):
                new_cost = cost(new_layout)
                delta_cost = new_cost - current_cost
                if delta_cost <= 0 or random.random() < np.exp(-(delta_cost + 1e-10) / T):
                    current_layout = new_layout
                    current_cost = new_cost
                    if changed == 1:
                        accepted_count += 1
            total_count += 1
            if current_cost < best_cost:
                best_layout = current_layout
                best_cost = current_cost
        acceptance_rate = accepted_count / total_count
        num_iterations += 1
        if acceptance_rate < target_acceptance - acceptance_window:
            alpha = max(alpha * (1 - alpha_factor), alpha_min)
        elif acceptance_rate > target_acceptance + acceptance_window:
            alpha = min(alpha * (1 + alpha_factor), alpha_max)
        T *= 1 - alpha
        temp.append(T)
        unchanged_iterations += 1
    # Plot temperature over time
    # plt.plot(temp)
    # plt.xlabel('Iteration')
    # plt.ylabel('Temperature')
    # plt.title('Temperature Decrease Over Time')
    # plt.show()
    return best_layout


#this Simulated Annealing function is used for comparing purposes. Here we run the loop for constant time and see the bounding area after that time
def simulated_annealing_timed(fpga_width, fpga_height, max_time, alpha_norm, T_init, T_min, alpha_min, alpha_max,
                              alpha_factor, inner_loop, target_acceptance, acceptance_window):
    start_time = time.time()
    T = T_init
    current_layout = generate_layout(fpga_width, fpga_height)
    current_cost = cost(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    temp = [T]
    while time.time() - start_time < max_time:
        unchanged_iterations = 0
        accepted_count = 0
        total_count = 0
        num_iterations = 0
        alpha = alpha_norm
        for i in range(inner_loop):
            new_layout, changed = generate_new_layout(current_layout, fpga_width, fpga_height)
            if check_validity(new_layout, fpga_width, fpga_height):
                new_cost = cost(new_layout)
                delta_cost = new_cost - current_cost
                if delta_cost <= 0 or random.random() < np.exp(-(delta_cost + 1e-10) / T):
                    current_layout = new_layout
                    current_cost = new_cost
                    if changed == 1:
                        accepted_count += 1
            total_count += 1
            if current_cost < best_cost:
                best_layout = current_layout
                best_cost = current_cost
        acceptance_rate = accepted_count / total_count
        num_iterations += 1
        if acceptance_rate < target_acceptance - acceptance_window:
            alpha = max(alpha * (1 - alpha_factor), alpha_min)
        elif acceptance_rate > target_acceptance + acceptance_window:
            alpha = min(alpha * (1 + alpha_factor), alpha_max)
        T *= alpha
        temp.append(T)
        unchanged_iterations += 1

    # Plot temperature over time
    # plt.plot(temp)
    # plt.xlabel('Iteration')
    # plt.ylabel('Temperature')
    # plt.title('Temperature Decrease Over Time')
    # plt.show()

    return best_layout


#and time.time() - start_time < max_time
def simulated_annealing_bounding_area(fpga_width, fpga_height, req_bounding_area, max_time, alpha_norm, T_init, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, target_acceptance, acceptance_window):
    T = T_init
    print('I am inside bounding SA')
    current_layout = generate_layout(fpga_width, fpga_height)
    current_cost = cost(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    unchanged_iterations = 0
    start_time = time.time()
    while bounding_area(current_layout) > req_bounding_area and unchanged_iterations < max_unchanged_iterations:
        #print(bounding_area(current_layout))
        unchanged_iterations = 0
        accepted_count = 0
        total_count = 0
        num_iterations = 0
        alpha = alpha_norm
        for i in range(inner_loop):
            new_layout, changed = generate_new_layout(current_layout, fpga_width, fpga_height)
            if check_validity(new_layout, fpga_width, fpga_height):
                new_cost = cost(new_layout)
                delta_cost = new_cost - current_cost
                if delta_cost <= 0 or random.random() < np.exp(-(delta_cost + 1e-10) / T):
                    current_layout = new_layout
                    current_cost = new_cost
                    if changed == 1:
                        accepted_count += 1
            total_count += 1
            if current_cost < best_cost:
                best_layout = current_layout
                best_cost = current_cost
        acceptance_rate = accepted_count / total_count
        num_iterations += 1
        if acceptance_rate < target_acceptance - acceptance_window:
            alpha = max(alpha * (1 - alpha_factor), alpha_min)
        elif acceptance_rate > target_acceptance + acceptance_window:
            alpha = min(alpha * (1 + alpha_factor), alpha_max)
        T *= 1 - alpha
        unchanged_iterations += 1
        #Check if the time limit has been reached
        if (time.time() - start_time) > max_time:
            break
    return best_layout

#

#
#get the file name and algorithm from the command line
if len(sys.argv) != 3:
    print("Usage: python your_script.py <file_name> <algorithm>")
    exit()

filename = sys.argv[1]
algorithm = sys.argv[2]

# Read input from the file
fpga_width, fpga_height, block_sizes = read_input_file(filename)
#fpga_width, fpga_height, block_sizes = read_experiment_file(filename)
if algorithm == "SA":
    #for running SA loop with T
    start_time = time.time()
    final_layout = simulated_annealing(fpga_width, fpga_height, T, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, max_unchanged_iterations, target_acceptance, acceptance_window)
    end_time = time.time()
    runtime = end_time - start_time
    print('runtime = ',runtime)
    final_bounding_area = bounding_area(final_layout)
    print('final_bounding_area = ',final_bounding_area)
    visualize(final_layout)
    print(len(final_layout))
elif algorithm == "SAT":
    # for running SA loop with constant time period
    max_time = 240

    final_layout_timed = simulated_annealing_timed(fpga_width, fpga_height, max_time, alpha_norm, T_init, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, target_acceptance, acceptance_window)
    final_bounding_area = bounding_area(final_layout_timed)
    print('final_bounding_area = ',final_bounding_area)
    visualize(final_layout_timed)
    print(len(final_layout_timed))
    for i in range(len(final_layout_timed)):
        print(final_layout_timed[i])
        print(final_layout_timed[i].get_label())
elif algorithm == "SAB":
    start_time = time.time()
    req_bounding_area = 165
    final_layout = simulated_annealing_bounding_area(fpga_width, fpga_height, req_bounding_area, alpha_norm, T_init, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, target_acceptance, acceptance_window)
    end_time = time.time()
    runtime = end_time - start_time
    print('runtime = ', runtime)
    final_bounding_area = bounding_area(final_layout)
    print('final_bounding_area = ', final_bounding_area)
    visualize(final_layout)
    print(len(final_layout))
else:
    print('invalid algorithm')

#
# df1 = pd.read_csv('optimal_LBMA_final.csv', delimiter=';', usecols=['filename', 'final_bounding_area', 'runtime'])
# #
# #this code for testing purposes
# folder_path = "../Test data/LBMA_Data"
# files = os.listdir(folder_path)
# # filter out the files that do not exist in df1
# filtered_files = [filename for filename in files if filename.endswith(".txt") and filename in df1['filename'].values]
#  # maximum number of attempts to reach the required bounding area
# #files = ['MBLA_1.txt','MBLA_10.txt','MBLA_11.txt','MBLA_16.txt','MBLA_17.txt','MBLA_2.txt','MBLA_20.txt','MBLA_21.txt','MBLA_26.txt','MBLA_28.txt','MBLA_3.txt','MBLA_31.txt','MBLA_32.txt','MBLA_33.txt','MBLA_34.txt','MBLA_37.txt','MBLA_38.txt','MBLA_39.txt','MBLA_4.txt','MBLA_43.txt','MBLA_48.txt','MBLA_49.txt','MBLA_50.txt','MBLA_51.txt','MBLA_53.txt','MBLA_59.txt','MBLA_7.txt']
# results = []
# max_attempts = 3
#
# with open('LBMA_SA_reach_optimal', 'w', newline='') as csvfile:
#     fieldnames = ['filename', 'final_bounding_area', 'width', 'runtime', 'total_blocks', 'final_num_blocks', 'num_attempts']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#
#     for filename in filtered_files:
#         if not filename.endswith(".txt"):
#             continue
#
#         print(filename)
#         filepath = os.path.join(folder_path, filename)
#         fpga_width, fpga_height, block_sizes = read_input_file(filepath)
#         max_time = 500
#         req_bounding_area = df1.loc[df1['filename'] == filename, 'final_bounding_area'].iloc[0]
#
#         min_bounding_area_result = None
#         attempt = 1
#         while attempt <= max_attempts:
#             start_time = time.time()
#             final_layout = simulated_annealing_bounding_area(fpga_width, fpga_height, req_bounding_area, max_time, alpha_norm, T_init, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, target_acceptance, acceptance_window)
#             end_time = time.time()
#
#             runtime = end_time - start_time
#
#             got_bounding_area, width = final_bounding_area(final_layout)
#
#             print(f"Filename {filename}, Final bounding area: {got_bounding_area}")
#
#             if got_bounding_area <= req_bounding_area:
#                 # Write to CSV and exit while loop
#                 results.append({
#                     'filename': filename,
#                     'final_bounding_area': got_bounding_area,
#                     'width': width,
#                     'runtime': runtime,
#                     'total_blocks': len(block_sizes),
#                     'final_num_blocks': len(final_layout),
#                     'num_attempts': attempt
#                 })
#                 writer.writerow(results[-1])
#                 break
#
#             # Otherwise, store the result in the list and keep iterating
#             if min_bounding_area_result is None or got_bounding_area < min_bounding_area_result['final_bounding_area']:
#                 min_bounding_area_result = {
#                     'filename': filename,
#                     'final_bounding_area': got_bounding_area,
#                     'width': width,
#                     'runtime': runtime,
#                     'total_blocks': len(block_sizes),
#                     'final_num_blocks': len(final_layout),
#                     'num_attempts': attempt
#                 }
#
#             if attempt == max_attempts:
#                 # Write the best result to CSV and exit while loop
#                 results.append({
#                     'filename': min_bounding_area_result['filename'],
#                     'final_bounding_area': min_bounding_area_result['final_bounding_area'],
#                     'width':min_bounding_area_result['width'],
#                     'runtime': min_bounding_area_result['runtime'],
#                     'total_blocks': len(block_sizes),
#                     'final_num_blocks': min_bounding_area_result['final_num_blocks'],
#                     'num_attempts': min_bounding_area_result['num_attempts']
#                 })
#                 writer.writerow(results[-1])
#
#             attempt += 1

# # #this code for testing purposes
# folder_path = "../Test data/LBMA_Data"
# files = os.listdir(folder_path)
# # filter out the files that do not exist in df1
# #filtered_files = [filename for filename in files if filename.endswith(".txt") and filename in df1['filename'].values]
# results = []
# max_attempts = 3  # maximum number of attempts to reach the required bounding area
# # files = ['MBLA_17.txt','MBLA_18.txt','MBLA_28.txt','MBLA_46.txt']
# for filename in files:
#     if filename.endswith(".txt"):
#         print(filename)
#         filepath = os.path.join(folder_path, filename)
#         fpga_width, fpga_height, block_sizes = read_input_file(filepath)
#         max_time = 360
#         start_time = time.time()
#         final_layout = simulated_annealing_timed(fpga_width, fpga_height, max_time, alpha_norm, T_init, T_min, alpha_min, alpha_max,alpha_factor, inner_loop, target_acceptance, acceptance_window)
#         end_time = time.time()
#         runtime = end_time - start_time
#         final_bounding_area = bounding_area(final_layout)
#         results.append({
#                 'filename': filename,
#                 'final_bounding_area': final_bounding_area,
#                 'runtime': runtime,
#                 'total_blocks': len(block_sizes),
#                 'final_num_blocks': len(final_layout)
#             })
#
#         print(f"Filename {filename}, Final bounding area: {final_bounding_area}")
#
#         # Write the results to a CSV file
#         with open('LBMA_timed_SA_6mins.csv', 'w', newline='') as csvfile:
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
#         #max_time = 120
#
#         min_bounding_area = float('inf')
#         min_result = None
#
#         for attempt in range(max_attempts):
#             start_time = time.time()
#             final_layout = simulated_annealing(fpga_width, fpga_height, T, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, max_unchanged_iterations, target_acceptance, acceptance_window)
#             end_time = time.time()
#
#             runtime = end_time - start_time
#             final_bounding_area = bounding_area(final_layout)
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
#                     'final_num_blocks': len(final_layout),
#                     'num_attempts': attempt + 1
#                 }
#
#         # Write the results to a CSV file
#         with open('final_experiment_file_SA_normal.csv', 'a', newline='') as csvfile:
#             fieldnames = ['filename', 'final_bounding_area', 'runtime', 'total_blocks', 'final_num_blocks',
#                           'num_attempts']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#             if os.stat('final_experiment_file_SA_normal.csv').st_size == 0:
#                 writer.writeheader()
#
#             writer.writerow(min_result)

#