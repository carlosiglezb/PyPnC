import numpy as np


def get_last_defined_point(safe_points_list, frame_name):
    for sp in reversed(safe_points_list):
        if frame_name in sp.keys():
            return sp[frame_name]

    # if we reach this point, the corresponding frame is never assigned
    return 0


def distribute_box_seq(box_seq_all_frames, b_max):
    for frame, seq in box_seq_all_frames.items():
        if not np.isnan(seq[0]) and len(seq) != b_max:
            # if all values are the same, simply copy up to b_max
            if np.mean(seq) == seq[-1]:
                box_seq_all_frames[frame] = [seq[-1]] * b_max
            else:
                # approach 1: repeat the last entry until we match the size b_max
                last_box = seq[-1]
                missing_entries = b_max - len(seq)
                box_seq_all_frames[frame] = seq + missing_entries * [last_box]


def get_last_defined_box(box_seq_list, frame_name):
    for bs in reversed(box_seq_list):
        if frame_name in bs.keys() and not any(np.isnan(bs[frame_name])):
            return bs[frame_name]

    # if we reach this point, the corresponding frame is never assigned
    print(f'Frame {frame_name} is never assigned in the box sequence list')
    return np.nan


def get_num_unassigned_boxes(box_seq_list, frame_name):
    num_unassigned_boxes = 1
    for bs in reversed(box_seq_list):
        if any(np.isnan(bs[frame_name])):
            num_unassigned_boxes += 1   # update location of last nan box
        else:
            return num_unassigned_boxes


def unassigned_box_seq_interpolator(box_seq_list, last_box_seq, frame_name):
    num_boxes_unassigned = get_num_unassigned_boxes(box_seq_list, frame_name)

    last_defined_box_seq = get_last_defined_box(box_seq_list, frame_name)

    # if there are no unassigned boxes, and the box seq list contains nan, over-write it
    if num_boxes_unassigned is None:
        if np.isnan(last_defined_box_seq):
            if last_box_seq[frame_name][0] is not None:
                last_defined_box_seq = [last_box_seq[frame_name][0]]
                num_boxes_unassigned = 1
                print('[warning] Double-check the box sequence list')

    # check last defined box sequence is consistent
    if last_defined_box_seq[0] != last_box_seq[frame_name][0]:
        print(f"[warning] Box sequence in {frame_name} free frame is inconsistent. "
              f"Check that seeds for {frame_name} frame are contained in both IRIS regions")

    interval_boxes = last_box_seq[frame_name][-1] - last_box_seq[frame_name][0]
    fract_box = interval_boxes / num_boxes_unassigned

    # distribute boxes equally
    k_box = 0
    for bs in box_seq_list:
        new_box_seq_val = round(last_defined_box_seq[0] + k_box * fract_box)
        b_max = np.max([len(boxes) for boxes in bs.values()])
        bs[frame_name] = [new_box_seq_val] * b_max
        k_box += 1

    # clear boxes from last box_seq
    last_box_seq[frame_name] = [last_box_seq[frame_name][-1]]


def distribute_free_frames(last_box_seq, box_seq_list, frame_name):

    # distribute according to the number of segments allocated
    unassigned_box_seq_interpolator(box_seq_list, last_box_seq, frame_name)
