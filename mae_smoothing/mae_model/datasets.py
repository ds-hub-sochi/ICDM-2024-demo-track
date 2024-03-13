import os
from kinetics import VideoClsDataset
import pandas as pd


def build_dataset(args):
    dataset = VideoClsDataset(
        anno_path=args.anno_path,
        anno_type=args.anno_type,
        data_path=args.data_path,
        mode='validation',
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        args=args,
    )
    nb_classes = len(set(dataset.label_array))
    print('Number of classes = %d' % nb_classes)
    return dataset, nb_classes
