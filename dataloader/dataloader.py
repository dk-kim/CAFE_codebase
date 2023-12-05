# ------------------------------------------------------------------------
# Modified from SSU (https://github.com/cvlab-epfl/social-scene-understanding)
# Modified from ARG (https://github.com/wjchaoGit/Group-Activity-Recognition)
# ------------------------------------------------------------------------
from .cafe import *

import pickle

TRAIN_CAFE_P = ['1', '2', '3', '4', '9', '10', '11', '12', '17', '18', '19', '20', '21', '22', '23', '24']
VAL_CAFE_P = ['13', '14', '15', '16']
TEST_CAFE_P = ['5', '6', '7', '8']

TRAIN_CAFE_V = ['1', '2', '5', '6', '9', '10', '13', '14', '17', '18', '21', '22']
VAL_CAFE_V = ['3', '7', '11', '15', '19', '23']
TEST_CAFE_V = ['4', '8', '12', '16', '20', '24']


def read_dataset(args):
    if args.dataset == 'cafe':
        data_path = args.data_path + 'cafe'

        # split-by-place setting
        if args.split == 'place':
            TRAIN_VIDEOS_CAFE = TRAIN_CAFE_P
            VAL_VIDEOS_CAFE = VAL_CAFE_P
            TEST_VIDEOS_CAFE = TEST_CAFE_P
        # split-by-view setting
        elif args.split == 'view':
            TRAIN_VIDEOS_CAFE = TRAIN_CAFE_V
            VAL_VIDEOS_CAFE = VAL_CAFE_V
            TEST_VIDEOS_CAFE = TEST_CAFE_V
        else:
            assert False

        if args.val_mode:
            train_data = cafe_read_annotations(data_path, TRAIN_VIDEOS_CAFE, args.num_class)
            train_frames = cafe_all_frames(train_data)

            test_data = cafe_read_annotations(data_path, VAL_VIDEOS_CAFE, args.num_class)
            test_frames = cafe_all_frames(test_data)
        else:
            train_data = cafe_read_annotations(data_path, TRAIN_VIDEOS_CAFE + VAL_VIDEOS_CAFE, args.num_class)
            train_frames = cafe_all_frames(train_data)

            test_data = cafe_read_annotations(data_path, TEST_VIDEOS_CAFE, args.num_class)
            test_frames = cafe_all_frames(test_data)

        # actor tracklets for all frames
        all_tracks = pickle.load(open(data_path + '/gt_tracks_24.pkl', 'rb'))

        train_set = CafeDataset(train_frames, train_data, all_tracks, data_path, args, is_training=True)
        test_set = CafeDataset(test_frames, test_data, all_tracks, data_path, args, is_training=False)
    else:
        assert False

    print("%d train samples and %d test samples" % (len(train_frames), len(test_frames)))

    return train_set, test_set
