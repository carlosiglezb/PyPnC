

def get_last_defined_point(safe_points_list, frame_name):
    for sp in reversed(safe_points_list):
        if frame_name in sp.keys():
            return sp[frame_name]

    # if we reach this point, the corresponding frame is never assigned
    return 0
