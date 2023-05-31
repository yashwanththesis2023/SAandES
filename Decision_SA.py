import random
import math
import numpy as np
import sympy
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import csv
import time
import sys


# generate intial layout
def generate_layout(fpga_width, fpga_height, block_sizes, preplaced_blocks):
    layout = []
    placed_blocks = set()
    # Place the preplaced block
    for preplaced_block in preplaced_blocks:
        rect = Rectangle((preplaced_block["x"], preplaced_block["y"]), preplaced_block["width"],
                         preplaced_block["height"], color=np.random.rand(3, ))
        rect.set_label(preplaced_block["name"])
        # layout.append(rect)
        placed_blocks.add(preplaced_block["name"])

    # Calculate the available micro slots after placing the pre-placed blocks
    available_micro_slots = fpga_width * fpga_height - sum(
        [preplaced_block["width"] * preplaced_block["height"] for preplaced_block in preplaced_blocks])
    # print(available_micro_slots)
    required_area = sum([block_size for block_size in block_sizes.values()])
    if required_area > available_micro_slots:
        print("Need a bigger FPGA because area requirement is more than available micro slots")
        return None
    sorted_blocks = sorted(block_sizes.items(), key=lambda x: x[1], reverse=True)
    max_attempts = 100  # set a maximum number of attempts to place a block
    for block, area in sorted_blocks:
        if block in placed_blocks:
            continue
        placed = False
        attempts = 0
        original_area = area
        if sympy.isprime(area) and (area > fpga_height): area = area + 1
        while not placed and attempts < max_attempts:
            x = random.randint(0, fpga_width)
            y = random.randint(0, fpga_height)
            # print('shape of the block',w,h)
            # print('x and y ',x,y)
            shapes = get_shape(area)
            w = shapes[0]
            h = shapes[1]
            rectangle = Rectangle((x, y), w, h, color=np.random.rand(3, ))
            rectangle.set_label(block)
            if check_validity(layout + [rectangle], fpga_width, fpga_height, preplaced_blocks):
                layout.append(rectangle)
                placed_blocks.add(block)
                placed = True
                # Update the available micro slots after placing the block
                available_micro_slots -= area
                break
            attempts += 1

        if attempts == max_attempts and not placed:
            print(f"Block {block} could not be placed.")
            print(f"need a bigger FPGA")

    if (len(layout) == (len(block_sizes))):
        packed = True
        return layout, packed
    else:
        packed = False
        return layout, packed


# get the shape (width, height) by providing area
def get_shape(area):
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
    # sort rectangles by perimeter (width + height)
    # rectangles.sort(key=lambda x: sum(x))
    block = random.choice(rectangles)
    return block


# get all possible rectangles from the given area
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
    # sort rectangles by perimeter (width + height)
    # rectangles.sort(key=lambda x: sum(x))
    # block = random.choice(rectangles)
    return rectangles


# check function to see the blocks are not placed out of FPGA and on preplaced blocks
def check_validity(layout, fpga_width, fpga_height, preplaced_blocks):
    # check for overlapping and boundaries
    coordinates = set()
    for rectangle in layout:
        x0, y0 = rectangle.get_xy()
        x1 = x0 + rectangle.get_width()
        y1 = y0 + rectangle.get_height()

        # check if block is outside the boundaries
        if x1 > fpga_width or y1 > fpga_height:
            return False

        # check if block overlaps with any preplaced blocks
        for preplaced_block in preplaced_blocks:
            px, py = preplaced_block['x'], preplaced_block['y']
            p_x = px + preplaced_block['width']
            p_y = py + preplaced_block['height']

            # check if block overlaps with preplaced block
            if x1 > px and y1 > py and x0 < p_x and y0 < p_y:
                return False

        # add block coordinates to set
        for x in range(int(x0), int(x1)):
            for y in range(int(y0), int(y1)):
                coordinates.add((x, y))

    # clear the coordinates set after all blocks have been checked
    coordinates.clear()
    return True


# this function calculates total overlaps between the blocks in given layout
def calc_overlap_cost(layout):
    overlap_count = 0
    for i, block1 in enumerate(layout):
        # print('inside i loop',block1)
        for j, block2 in enumerate(layout[i + 1:], i + 1):
            # print(block2)
            x_min = max(block1.get_xy()[0], block2.get_xy()[0])
            x_max = min(block1.get_xy()[0] + block1.get_width(), block2.get_xy()[0] + block2.get_width())
            y_min = max(block1.get_xy()[1], block2.get_xy()[1])
            y_max = min(block1.get_xy()[1] + block1.get_height(), block2.get_xy()[1] + block2.get_height())
            if x_min < x_max and y_min < y_max:
                overlap_count += 1
    return overlap_count


# changes the shape of the given block and gets new x and y positions
def change_shape(block, layout):
    shape_of_block = (block.get_width(), block.get_height())
    # print('current shape of the block',shape_of_block)
    shapes = get_rectangles(block.get_width() * block.get_height())
    # print('shapes',shapes)
    for w, h in shapes:
        if (w, h) != shape_of_block:
            # print('other shape of the block',w,h)
            block.set_width(w)
            block.set_height(h)
            changed = change_location(block, layout, fpga_width, fpga_height)
            if changed:
                return changed
    block.set_width(shape_of_block[0])
    block.set_height(shape_of_block[1])
    # print('couldnt find new loc in shape')
    changed = False
    return changed


# this functions finds the new x and y for the given block
max_iterations = 1000
def change_location(block, layout, fpga_width, fpga_height):
    x_old, y_old = block.get_xy()
    changed = False
    for _ in range(max_iterations):
        x = random.randint(0, fpga_width - 1)
        y = random.randint(0, fpga_height - 1)
        if (x, y) != (x_old, y_old):
            block.set_xy((x, y))
            if check_validity(layout, fpga_width, fpga_height, preplaced_blocks):
                changed = True
                break
    if not changed:
        block.set_xy((x_old, y_old))
    return changed


# this function is used to generate new layouts by changing the shape or changing the location
def generate_new_layout(layout, fpga_width, fpga_height):
    new_layout = layout.copy()
    block = random.choice(new_layout)
    if random.uniform(0, 1) < 0.5:
        # print('in changing')
        changed = change_shape(block, new_layout)
    else:
        changed = change_location(block, new_layout, fpga_width, fpga_height)
    return new_layout, changed


# this function reads the file and gets fpga_width, fpga_height, block_sizes
def read_input_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    outline = data[0].strip().split()
    fpga_width, fpga_height = int(outline[1]), int(outline[2])
    block_sizes = {}
    for line in data[1:]:
        # print(line.strip().split())
        block, size = line.strip().split()
        block_sizes[block] = int(size)
    return fpga_width, fpga_height, block_sizes


# visualizes the layout
def visualize(layout, preplaced_blocks):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Compute the area of each rectangle
    areas = [rectangle.get_width() * rectangle.get_height() for rectangle in layout]

    # Normalize the areas to be between 0 and 1
    normalized_areas = [(area - min(areas)) / (max(areas) - min(areas)) for area in areas]

    # Create a colormap
    cmap = cm.get_cmap("viridis")

    for preplaced_block in preplaced_blocks:
        if preplaced_block["name"].startswith("preplaced"):
            color = "black"
            rect = plt.Rectangle((preplaced_block["x"], preplaced_block["y"]), preplaced_block["width"],
                                 preplaced_block["height"],
                                 facecolor=color, fill=True, edgecolor='black')
            ax.add_patch(rect)
    for i, rectangle in enumerate(layout):
        color = cmap(normalized_areas[i])
        rect = plt.Rectangle((rectangle.get_xy()), rectangle.get_width(), rectangle.get_height(),
                             facecolor=color, fill=True, edgecolor='black')
        ax.add_patch(rect)

        # Find the corresponding name of the rectangle
        # Add the block name and area to the center of the patch
        ax.annotate("{}\n{:.2f}".format(rectangle.get_label(), areas[i]), (
            rectangle.get_xy()[0] + rectangle.get_width() / 2, rectangle.get_xy()[1] + rectangle.get_height() / 2),
                    color='black', fontsize=18, ha='center', va='center')
    # Set the xticks and yticks
    ax.set_xticks(range(fpga_width + 1))
    ax.set_yticks(range(fpga_height + 1))

    # Set the yticks label format to integers
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: int(val)))
    ax.grid(True, which='both', color='grey', linewidth=1.0)
    plt.xlabel('fpga-width',fontsize=20)
    plt.ylabel('fpga-height',fontsize=20)
    plt.xlim(0, fpga_width)
    plt.ylim(0, fpga_height)
    plt.savefig('final_decision_SA_sample')
    plt.show()


# required parameters for running SA function

T = 100000000
T_min = 0.00001
alpha_min = 0.001
alpha_norm = 0.3
alpha_max = 0.7
alpha_factor = 0.1
inner_loop = 100
max_unchanged_iterations = 1000
target_acceptance = 0.5
acceptance_window = 0.1


def simulated_annealing(fpga_width, fpga_height, block_sizes, preplaced_blocks, T, T_min, alpha_norm, alpha_min,
                        alpha_max, alpha_factor, inner_loop, target_acceptance, acceptance_window):
    current_layout, placed = generate_layout(fpga_width, fpga_height, block_sizes, preplaced_blocks)
    if not placed:
        return None
    current_cost = calc_overlap_cost(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    unchanged_iterations = 0
    while T > T_min and best_cost != 0:
        current_unchanged_iterations = 0
        accepted_count = 0
        total_count = 0
        alpha = alpha_norm
        for i in range(inner_loop):
            new_layout, changed = generate_new_layout(current_layout, fpga_width, fpga_height)
            if check_validity(new_layout, fpga_width, fpga_height, preplaced_blocks):
                new_cost = calc_overlap_cost(new_layout)
                delta_cost = new_cost - current_cost
                if delta_cost <= 0 or random.random() < np.exp(-(delta_cost + 1e-10) / T):
                    current_layout = new_layout
                    current_cost = new_cost
                    if current_cost == 0:
                        return current_layout
                    if changed == 1:
                        accepted_count += 1
                total_count += 1
                if current_cost < best_cost:
                    best_layout = current_layout
                    best_cost = current_cost
                    if best_cost == 0:
                        return best_layout
        acceptance_rate = accepted_count / total_count
        if acceptance_rate < target_acceptance - acceptance_window:
            alpha = max(alpha * (1 - alpha_factor), alpha_min)
        elif acceptance_rate > target_acceptance + acceptance_window:
            alpha = min(alpha * (1 + alpha_factor), alpha_max)
        T *= 1 - alpha
        unchanged_iterations += 1
    return best_layout


# this tells the final layout has all given blocks or not
def final_accept(layout):
    if layout != None:
        final_cost = calc_overlap_cost(layout)
        if final_cost == 0:
            return True
        else:
            return False
    else:
        return False


# filename = "..\Test Data\Final_decision_data\Zync7200_4.txt"
# fpga_width, fpga_height, block_sizes = read_input_file(filename)
# preplaced_blocks = [{"name": "preplaced_block1", "x": 0, "y": 1, "width": 2, "height": 2},
#                     {"name": "preplaced_block2", "x": 3, "y": 0, "width": 1, "height": 3}]
#
# final_layout = simulated_annealing(fpga_width, fpga_height, block_sizes, preplaced_blocks, T, T_min, alpha_norm,
#                                    alpha_min, alpha_max, alpha_factor, inner_loop, target_acceptance, acceptance_window)
# final_acceptance = final_accept(final_layout)
# print(final_acceptance)
# visualize(final_layout, preplaced_blocks)

#get the file name and algorithm from the command line
if len(sys.argv) != 3:
    print("Usage: python your_script.py <file_name> <device>")
    exit()

filename = sys.argv[1]
device = sys.argv[2]
#Read input from the file
fpga_width, fpga_height, block_sizes = read_input_file(filename)
if device == "Zynq7020":
    preplaced_blocks = [{"name": "preplaced_block1", "x": 0, "y": 1, "width": 2, "height": 2},
                        {"name": "preplaced_block2", "x": 3, "y": 0, "width": 1, "height": 3}]

    final_layout = simulated_annealing(fpga_width, fpga_height, block_sizes, preplaced_blocks, T, T_min, alpha_norm,
                                       alpha_min, alpha_max, alpha_factor, inner_loop, target_acceptance,
                                       acceptance_window)
    final_acceptance = final_accept(final_layout)
    print(final_acceptance)
    visualize(final_layout, preplaced_blocks)

if device == "Zynq7030":
    preplaced_blocks = [{"name": "preplaced_block1", "x": 0, "y": 2, "width": 2, "height": 2},
                        {"name": "preplaced_block2", "x": 4, "y": 0, "width": 1, "height": 4},
                        {"name": "preplaced_block3", "x": 6, "y": 0, "width": 1, "height": 1}]
    final_layout = simulated_annealing(fpga_width, fpga_height, block_sizes, preplaced_blocks, T, T_min, alpha_norm,
                                       alpha_min, alpha_max, alpha_factor, inner_loop, target_acceptance,
                                       acceptance_window)
    final_acceptance = final_accept(final_layout)
    print(final_acceptance)
    visualize(final_layout, preplaced_blocks)


# folder_path = "../Test data/Final_decision_data"
# files = os.listdir(folder_path)
# results = []
#
# for filename in files:
#     if filename.endswith(".txt"):
#         print(filename)
#         filepath = os.path.join(folder_path, filename)
#         fpga_width, fpga_height, blocks = read_input_file(filepath)
#
#         # print(population_size)
#         preplaced_blocks = [{"name": "preplaced_block1", "x": 0, "y": 1, "width": 2, "height": 2},
#                             {"name": "preplaced_block2", "x": 3, "y": 0, "width": 1, "height": 3}]
#
#         max_iterations = 10
#         iterations = 0
#         final_acceptance = False
#         while not final_acceptance and iterations < max_iterations:
#             start_time = time.time()
#             final_layout = simulated_annealing(fpga_width, fpga_height, blocks, preplaced_blocks, T, T_min, alpha_norm,
#                                        alpha_min, alpha_max, alpha_factor, inner_loop, target_acceptance,
#                                        acceptance_window)
#             end_time = time.time()
#
#             runtime = end_time - start_time
#             final_acceptance = final_accept(final_layout)
#             print(final_acceptance)
#
#             iterations += 1
#
#         # Store the results for this file
#         results.append({
#             'filename': filename,
#             'final_acceptance': final_acceptance,
#             'runtime': runtime
#         })
#
# # Write the results to a CSV file
# with open('Final_Decision_SA_results_new.csv', 'w', newline='') as csvfile:
#     fieldnames = ['filename', 'final_acceptance', 'runtime']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     writer.writeheader()
#     for result in results:
#         writer.writerow(result)
