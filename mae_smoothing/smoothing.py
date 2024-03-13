import argparse
import logging
import math
import os
import random
from collections import defaultdict

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_sim_method', default='rnd_smpl', choices=['rnd_smpl', 'mean_embeds'])
    parser.add_argument(
        '--data_path',
        help='path folder containing embeddings, labels and label encoder .npy files',
    )
    parser.add_argument('--betta', type=float, default=0.5, help='betta parameter value for class similarities')
    return parser.parse_args()


def get_mean_cos_matrix(embeds, labels, num_classes):  # noqa: WPS231, WPS210
    class_embeds = np.zeros((num_classes, 1024))
    class_counts = np.zeros(num_classes)
    for ind, label in enumerate(labels):
        class_embeds[label] += embeds[ind, :]
        class_counts[label] += 1
    for class_id in range(num_classes):
        if class_counts[class_id] > 0:
            class_embeds[class_id] = class_embeds[class_id] / class_counts[class_id]

    cosine_sim = np.zeros((num_classes, num_classes))
    for i in range(num_classes):  # noqa: WPS111
        for j in range(num_classes):  # noqa: WPS111
            if (class_counts[i]) != 0 and (class_counts[j] != 0) and i != j and i > j:  # noqa: WPS221
                cosine_sim[i, j] = np.dot(class_embeds[i], class_embeds[j]) / (  # noqa: WPS221
                    norm(class_embeds[i]) * norm(class_embeds[j])
                )
                cosine_sim[j, i] = cosine_sim[i, j]
    return cosine_sim


def save_most_similar_classes(sim_matrix, class_dict, output_path, top_n=50):
    ind = np.unravel_index(np.argsort(sim_matrix, axis=None), sim_matrix.shape)
    class_file_path = os.path.join(output_path, 'closest_classes.txt')
    with open(class_file_path, 'w') as class_file:
        for label_one, label_two in reversed(list(zip(ind[0], ind[1]))[-top_n:]):  # noqa: WPS221
            dist = sim_matrix[label_one, label_two]
            class_file.write(
                '{0} {1} {2:.2}\n'.format(
                    class_dict[label_one],
                    class_dict[label_two],
                    dist,
                ),  # noqa: WPS221, WPS221
            )


def get_embeds_dict(embeds, labels):
    embeds_dict = defaultdict(list)
    for index, label in enumerate(labels):
        embeds_dict[label].append(embeds[index])
    return embeds_dict


def calc_class_dist(embeds1, embeds2, per_class_num=10):
    if per_class_num < len(embeds1):
        embeds1 = random.sample(embeds1, per_class_num)
    if per_class_num < len(embeds2):
        embeds2 = random.sample(embeds2, per_class_num)
    dist = 0
    for emb_x in embeds1:
        for emb_y in embeds2:  # noqa: WPS519
            dist += norm(emb_x - emb_y)
    return dist / (len(embeds1) * len(embeds2))


def get_class_dist_matrix(embeds_dict, num_classes):
    dist_matrix = np.zeros((num_classes, num_classes))
    logging.info('Calculating class distance matrix')
    for i in tqdm(range(num_classes)):  # noqa: WPS111
        for j in range(i + 1, num_classes):  # noqa: WPS111
            dist_matrix[i, j] = calc_class_dist(embeds_dict[i], embeds_dict[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix


def calc_class_sim_matrix(dist_matrix, betta):  # it's going to be non-symmetrical
    num_classes = dist_matrix.shape[0]
    sim_matrix = np.zeros((num_classes, num_classes))
    logging.info('Calculating class similarity matrix')
    for label1 in tqdm(range(num_classes)):
        for label2 in range(num_classes):
            if label1 != label2:
                sim_matrix[label1, label2] = math.exp(-betta * dist_matrix[label1, label2])  # noqa: WPS221
        sim_matrix[label1, :] = sim_matrix[label1, :] / sum(sim_matrix[label1, :])  # noqa: WPS221
    return sim_matrix


def get_smoothed_labels(class_sim_matrix, alpha=0.1):
    num_classes = class_sim_matrix.shape[0]
    smth_labels = np.zeros((num_classes, num_classes))
    for i in range(num_classes):  # noqa: WPS111
        smth_labels[i, i] = 1
    return smth_labels * (1 - alpha) + class_sim_matrix * alpha


def read_class_dict(class_list_path):
    class_dict = {}
    with open(class_list_path, 'r') as class_list_file:
        lines = class_list_file.readlines()
    for line in lines:
        token_list = line.split()
        class_name = ' '.join(token_list[1:])
        class_name = class_name.strip('\n').strip(' ')
        class_label = int(token_list[0])
        class_dict[class_label] = class_name
    return class_dict


if __name__ == '__main__':
    args = get_args()
    logging.info(args)
    embeds = np.load(os.path.join(args.data_path, 'embeds.npy'))
    labels = np.load(os.path.join(args.data_path, 'labels.npy'))
    num_classes = len(set(labels))
    embeds_dict = get_embeds_dict(embeds, labels)
    if args.class_sim_method == 'rnd_smpl':
        dist_matrix = get_class_dist_matrix(embeds_dict, num_classes)
        sim_matrix = calc_class_sim_matrix(dist_matrix, args.betta)
    if args.class_sim_method == 'mean_embeds':
        sim_matrix = get_mean_cos_matrix(embeds, labels, num_classes)

    smoothed_labels = get_smoothed_labels(sim_matrix)
    save_path = os.path.join(args.data_path, 'soft_targets.npy')
    np.save(save_path, smoothed_labels)

    if 'class_list.txt' in os.listdir(args.data_path):
        class_dict = read_class_dict(os.path.join(args.data_path, 'class_list.txt'))
        save_most_similar_classes(sim_matrix, class_dict, args.data_path)  # noqa: WPS432
    else:
        logging.info('No class_list.txt was found inside specified data path')
