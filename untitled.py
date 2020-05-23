import pyautogui
import numpy as np
import time
import keyboard
from numba import njit

color_dict = {
    (0, 0, 0): 0,  # blank
    (15, 155, 215): 1,  # line
    (227, 159, 2): 2,  # square
    (175, 41, 138): 3,  # T
    (89, 177, 1): 4,  # S
    (215, 15, 55): 5,  # Z
    (33, 65, 198): 6,  # ReverseL
    (227, 91, 2): 7,  # L
    # any outline        : -1,
}

num_to_shape = {
    1: 'I',
    2: 'O',
    3: 'T',
    4: 'S',
    5: 'Z',
    6: 'J',
    7: 'L',
}


def colors_to_blocks(img):
    mat = np.ndarray(img.shape[:-1])
    for row_i, row in enumerate(img):
        for col_i, pixel in enumerate(row):
            scaled = pixel
            key = tuple(np.round(scaled).astype(np.int))
            if key in color_dict:
                mat[row_i, col_i] = color_dict[key]
            else:
                mat[row_i, col_i] = -1
    return mat


def colors_to_binary(img):
    mat = np.ndarray(img.shape[:-1])
    for row_i, row in enumerate(img):
        for col_i, pixel in enumerate(row):
            if tuple(pixel) == (0., 0., 0.):
                mat[row_i, col_i] = 0
            else:
                mat[row_i, col_i] = 1
    return mat


def screenshot():
    board = pyautogui.screenshot(region=(227, 183, 832-591, 652-173))
    board = np.array(board)

    queue = pyautogui.screenshot(region=(495, 183, 96, 360))
    queue = np.array(queue)

    board, queue = board[12::24, 12::24], queue[12::24, 12::24]

    board = colors_to_binary(board).astype(np.int)
    queue = colors_to_blocks(queue).astype(np.int)
    queue = [np.max(queue[0:3, 1:-1]),
             np.max(queue[3:6, 1:-1]),
             np.max(queue[6:9, 1:-1]),
             np.max(queue[9:12, 1:-1]),
             np.max(queue[12:15, 1:-1])]
    return board[3:], queue


protpos = dict()

protpos[('T', 0)] = ((1, 0), (0, 1), (1, 1), (1, 2))
protpos[('T', 3)] = ((2, 1), (1, 0), (1, 1), (0, 1))
protpos[('T', 2)] = ((1, 1), (0, 0), (0, 1), (0, 2))
protpos[('T', 1)] = ((2, 0), (1, 0), (0, 0), (1, 1))

protpos[('I', 0)] = ((0, 0), (0, 1), (0, 2), (0, 3))
protpos[('I', 1)] = ((3, 0), (1, 0), (2, 0), (0, 0))

protpos[('O', 0)] = ((1, 0), (0, 0), (0, 1), (1, 1))

protpos[('S', 0)] = ((1, 0), (0, 1), (1, 1), (0, 2))
protpos[('S', 1)] = ((2, 1), (1, 0), (1, 1), (0, 0))

protpos[('Z', 0)] = ((1, 1), (0, 0), (0, 1), (1, 2))
protpos[('Z', 1)] = ((2, 0), (1, 0), (0, 1), (1, 1))

protpos[('J', 0)] = ((1, 0), (0, 0), (1, 1), (1, 2))
protpos[('J', 3)] = ((2, 0), (0, 1), (1, 1), (2, 1))
protpos[('J', 2)] = ((0, 0), (0, 1), (0, 2), (1, 2))
protpos[('J', 1)] = ((2, 0), (1, 0), (0, 0), (0, 1))

protpos[('L', 0)] = ((1, 0), (0, 2), (1, 1), (1, 2))
protpos[('L', 3)] = ((2, 1), (0, 0), (1, 1), (0, 1))
protpos[('L', 2)] = ((0, 0), (0, 1), (0, 2), (1, 0))
protpos[('L', 1)] = ((2, 0), (1, 0), (0, 0), (2, 1))


poff = dict()

poff[('T', 0)] = (3, 4)
poff[('T', 1)] = (4, 5)
poff[('T', 2)] = (3, 4)
poff[('T', 3)] = (3, 4)

poff[('I', 0)] = (3, 3)
poff[('I', 1)] = (5, 5)

poff[('O', 0)] = (4, 4)

poff[('S', 0)] = (3, 4)
poff[('S', 1)] = (4, 5)

poff[('Z', 0)] = (3, 4)
poff[('Z', 1)] = (4, 5)

poff[('J', 0)] = (3, 4)
poff[('J', 1)] = (4, 5)
poff[('J', 2)] = (3, 4)
poff[('J', 3)] = (3, 4)

poff[('L', 0)] = (3, 4)
poff[('L', 1)] = (4, 5)
poff[('L', 2)] = (3, 4)
poff[('L', 3)] = (3, 4)


@njit
def update_heights(fake_mat, block, drop_dist, drop_col):
    fake_copy = fake_mat.copy()
    for square in block:
        if square[1] + drop_col < 10:
            fake_copy[int(square[0] + drop_dist) - 1, int(square[1] + drop_col)] = 1

    return np.sum(fake_copy, axis=0)


@njit
def fake_from_heights(heights):
    retmat = np.ones((20, 10))
    for i, col in enumerate(heights):
        # print(i, col)
        retmat[:, i] = np.zeros(20)
        retmat[:int(col), i] = 1

    return retmat[::-1, :]


def calc_score(block_name, col_heights):
    num_orientations = 4
    if block_name == 'I' or block_name == 'S' or block_name == 'Z':
        num_orientations = 2
    elif block_name == 'O':
        num_orientations = 1

    fake_mat = fake_from_heights(col_heights)

    scores = np.empty((num_orientations, 10))
    drop_dists = np.empty((num_orientations, 10))

    for pos_col in range(10):
        # print(pos_col)
        for orientation in range(num_orientations):
            new_col_heights = col_heights.copy()
            block = protpos[(block_name, orientation)]

            num_holes = 0
            drop_dist = 0

            col_mins = dict()

            for square in block:
                tsquare = (square[0], square[1] + pos_col)
                if tsquare[1] < 10:
                    square_max_drop = (20 - tsquare[0]) - col_heights[tsquare[1]]
                    # print(square_max_drop, square[0], square[1])

                    if tsquare[1] in col_mins:
                        if square_max_drop < col_mins[tsquare[1]]:
                            col_mins[tsquare[1]] = square_max_drop
                    else:
                        col_mins[tsquare[1]] = square_max_drop
                else:
                    col_mins[tsquare[1]] = -1

            # print(col_mins)
            # print("----------")

            drop_dist = min(list(col_mins.values()))

            drop_dists[orientation, pos_col] = drop_dist

            fake_copy = fake_mat.copy()

            for square in block:
                if square[1] + pos_col < 10:
                    fake_copy[int(square[0] + drop_dist) - 1, int(square[1] + pos_col)] = 1

            new_col_heights = np.sum(fake_copy, axis=0)

            curr_holes = 0

            for item in col_mins.values():
                if item > drop_dist:
                    curr_holes += abs(item - drop_dist)

            num_holes += curr_holes

            prev_distr = np.max(col_heights) - np.min(col_heights)

            distr = np.max(new_col_heights) - np.min(new_col_heights)

            distr_diff = distr - prev_distr

            hole_fac = num_holes * 11
            if hole_fac > 12:
                hole_fac = 12

            score = (drop_dist / 1.6) - hole_fac - distr_diff

            #                 if not (block_name != 'I' and distr_diff > 1):
            #                     score -= distr_diff

            if num_holes == 0:
                score += 5

            max_h_diff = np.max(new_col_heights) - np.max(col_heights)
            if max_h_diff > 2 and block_name != 'I':
                score -= 2 * (max_h_diff - 1)

            if max_h_diff < 3 and num_holes == 0 and orientation == 1 and block_name == 'I':
                score += 15

            if block_name == 'I' and orientation == 0:
                score -= 5

            #             if pos_col >= 7 and block_name != 'I':
            #                 score -= 2 * pos_col - 6
            if block_name != 'I':
                for square in block:
                    if square[1] == 9:
                        score -= 10

            elif distr > 4:
                score -= distr

            prev_height = 25
            for col_height in new_col_heights:
                if col_height == prev_height and block_name != 'I':
                    score += 2
                elif abs(col_height - prev_height) > 3:
                    score -= abs(col_height - prev_height) * 1.2
                if col_height > 10:
                    score -= 2 * (col_height - 10) ** 3
                prev_height = col_height
            # print(col_heights, block_name, orientation, pos_col, drop_dist, num_holes, score)
            scores[orientation, pos_col] = score

    argmax = np.argmax(scores)
    drop_orientation = int(argmax // 10)
    drop_col = int(argmax % 10)

    block = protpos[(block_name, drop_orientation)]

    height_diffs = dict()

    new_col_heights = update_heights(fake_mat, block, drop_dists[drop_orientation, drop_col], drop_col)

    #     print(max_y)
    #     print(height_diffs)

    #     print(height_diffs)
    #     print("----")
    #     print(col_heights)
    #     print(block_name, drop_orientation, drop_col)
    #     print(new_col_heights)
    #     print('-------')

    return scores[drop_orientation, drop_col], drop_orientation, drop_col, new_col_heights


def main():
    time.sleep(3)
    # sum the columns to get max drop distance
    # if drop distance is more for one piece than for others, a hole will form

    board, queue = screenshot()
    current_block = queue[1]
    cache = queue[0]

    keyboard.press_and_release('space')
    keyboard.press_and_release('c')
    keydel = 1/64

    for frame in range(50_000):
        time.sleep(keydel)
        board, queue = screenshot()
        col_heights = np.zeros(10)
        for col in range(10):
            for row in range(len(board)):
                if board[row, col] != 0:
                    col_heights[col] = 17 - row
                    break

        block_name = num_to_shape[current_block]

        score, drop_orientation, drop_col, ncol_heights = calc_score(block_name, col_heights)

        cache_score, cdrop_orientation, cdrop_col, cncol_heights = calc_score(num_to_shape[cache], col_heights)
        if cache_score > score:
            drop_orientation = cdrop_orientation
            drop_col = cdrop_col
            block_name = num_to_shape[cache]
            keyboard.press_and_release('c')
            cache = int(current_block)
            ncol_heights = cncol_heights

            # print(score, cache_score)

        #     col_heights = ncol_heights
        #     print(col_heights)

        if drop_orientation == 3:
            keyboard.press_and_release('z')
        else:
            for i in range(drop_orientation):
                keyboard.press_and_release('up')

        offset = poff[(block_name, drop_orientation)][0] - drop_col

        if offset > 0:
            for _ in range(offset):
                keyboard.press_and_release('left')
        elif offset < 0:
            for _ in range(abs(offset)):
                keyboard.press_and_release('right')

        keyboard.press_and_release('space')
        current_block = queue[0]


main()
