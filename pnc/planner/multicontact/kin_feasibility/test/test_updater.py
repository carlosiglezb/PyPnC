frdict = None

def update_dict(frd):
    global frdict
    frdict = frd
    print("DICT UPDATED FLAG")

# update pos test
def reach_updater(reg_dict, frame_names, pos, op):
    if op is False:
        for fr in frame_names:
            reg_dict[fr].update_origin_pose(pos)
    else:
        for fr in frame_names:
            frdict[fr].update_origin_pose(pos)