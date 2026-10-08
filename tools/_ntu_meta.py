"""
NTU-RGB+D-25 joint metadata for visualisation.
"""

JOINT_NAMES_25 = [
    'spine base', 'spine mid', 'neck', 'head',
    'L shoulder', 'L elbow', 'L wrist', 'L hand',
    'R shoulder', 'R elbow', 'R wrist', 'R hand',
    'L hip', 'L knee', 'L ankle', 'L foot',
    'R hip', 'R knee', 'R ankle', 'R foot',
    'spine shoulder', 'L tip', 'L thumb', 'R tip', 'R thumb',
]

PART_OF_JOINT = {
    'head':      [2, 3, 20],
    'torso':     [0, 1],
    'left_arm':  [4, 5, 6, 7, 21, 22],
    'right_arm': [8, 9, 10, 11, 23, 24],
    'left_leg':  [12, 13, 14, 15],
    'right_leg': [16, 17, 18, 19],
}


def joint_to_part_color(palette):
    """Return a list of 25 colors mapped from joint index to body part."""
    out = ['#888888'] * 25
    for part, joints in PART_OF_JOINT.items():
        for j in joints:
            out[j] = palette[part]
    return out
