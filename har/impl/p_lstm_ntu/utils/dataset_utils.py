from random import randrange

import numpy as np


def get_all_body_parts_steps(data, analysed_body_parts, analysed_kpts_description, begin, end):
    body_parts = [body_part_steps(data, analysed_kpts_description[bp], begin, end) for bp in analysed_body_parts]
    return dict(zip(analysed_body_parts, body_parts))


def get_all_body_parts_splits(data, analysed_body_parts, analysed_kpts_description, split):
    body_parts = [body_part_splits(data, analysed_kpts_description[bp], split) for bp in analysed_body_parts]
    return dict(zip(analysed_body_parts, body_parts))


def body_part_steps(data, analysed_kpt_id, begin, end):
    return data[begin:end, analysed_kpt_id, :]


def body_part_splits(data, analysed_kpt_id, split):
    return np.array([a[randrange(len(a))] for a in np.array_split(data[:, analysed_kpt_id, :], split)])
