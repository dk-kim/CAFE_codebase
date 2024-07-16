import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import json
import numpy as np
import random
from PIL import Image

ACTIVITIES = ['Queueing', 'Ordering', 'Eating/Drinking', 'Working/Studying', 'Fighting', 'TakingSelfie']


# read annotation files
def cafe_read_annotations(path, videos, num_class):
    labels = {}
    group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}

    for vid in videos:
        video_path = os.path.join(path, vid)
        for cid in os.listdir(video_path):
            clip_path = os.path.join(video_path, cid)            
            label_path = clip_path + '/ann.json'

            with open(label_path, 'r') as file:
                groups = {}
                boxes, actions, activities, members, membership = [], [], [], [], []

                values = json.load(file)
                num_frames = values['framesCount']
                frame_interval = values['framesEach']
                actors = values['figures']

                key_frame = actors[0]['shapes'][0]['frame']

                for i, actor in enumerate(actors):
                    actor_idx = actor['id']
                    group_name = actor['label']

                    box = actor['shapes'][0]['coordinates']
                    x1, y1 = box[0]
                    x2, y2 = box[1]
                    x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    boxes.append([x_c, y_c, w, h])

                    if group_name != 'individual':
                        group_idx = int(group_name[-1])
                        if actor['attributes'][0]['value'] != "":
                            action = group_to_id[actor['attributes'][0]['value']['key']]

                            if group_idx not in groups.keys():
                                groups[group_idx] = {'activity': action}

                            if 'members' in groups[group_idx].keys():
                                groups[group_idx]['members'][i] = 1
                            else:
                                groups[group_idx]['members'] = torch.zeros(len(actors))
                                groups[group_idx]['members'][i] = 1
                        else:
                            if group_idx in groups.keys():
                                action = groups[group_idx]['activity']
                            else:
                                action = -1
                    else:
                        action = num_class
                        group_idx = 0

                    actions.append(action)
                    membership.append(group_idx)

                for i, action in enumerate(actions):
                    if action == -1:
                        group_idx = membership[i]

                        if group_idx in groups.keys():
                            new_action = groups[group_idx]['activity']
                            actions[i] = new_action

                            group_members = groups[group_idx]['members']
                            group_members[i] = 1
                        else:
                            membership[i] = 0
                            actions[i] = num_class

                for group_id in sorted(groups):
                    if group_id - 1 >= len(groups):
                        new_id = len(groups)

                        while new_id > 0:
                            if new_id not in groups:
                                groups[new_id] = groups[group_id]
                                del groups[group_id]
                                for i in range(len(membership)):
                                    if membership[i] == group_id:
                                        membership[i] = new_id
                                group_id = new_id
                            new_id -= 1

                for group_id in sorted(groups):
                    activities.append(groups[group_id]['activity'])
                    members.append(groups[group_id]['members'])

                actions = np.array(actions, dtype=np.int32)
                boxes = np.vstack(boxes)
                membership = np.array(membership, dtype=np.int32) - 1
                activities = np.array(activities, dtype=np.int32)

                actions = torch.from_numpy(actions).long()
                boxes = torch.from_numpy(boxes).float()
                membership = torch.from_numpy(membership).long()
                activities = torch.from_numpy(activities).long()

                if len(members) == 0:
                    members = torch.tensor(members)
                else:
                    members = torch.stack(members).float()

                annotations = {
                    'boxes': boxes,
                    'actions': actions,
                    'membership': membership,
                    'activities': activities,
                    'members': members,
                    'num_frames': num_frames,
                    'interval': frame_interval,
                    'key_frame': key_frame,
                }

            if len(annotations['activities']) != 0:
                labels[(int(vid), int(cid))] = annotations

    return labels


def cafe_all_frames(labels):
    frames = []

    for sid, anns in labels.items():
        frames.append(sid)
    return frames


class CafeDataset(data.Dataset):
    def __init__(self, frames, anns, tracks, image_path, args, is_training=True):
        super(CafeDataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.tracks = tracks
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.num_boxes = args.num_boxes
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_class = args.num_class
        self.is_training = is_training
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        if self.num_frame == 1:
            frames = self.select_key_frames(self.frames[idx])
        else:
            frames = self.select_frames(self.frames[idx])

        samples = self.load_samples(frames)

        return samples

    def __len__(self):
        return len(self.frames)

    def select_key_frames(self, frame):
        annotation = self.anns[frame]
        key_frame = annotation['key_frame']

        return [(frame, int(key_frame))]

    def select_frames(self, frame):
        annotation = self.anns[frame]
        key_frame = annotation['key_frame']
        total_frames = annotation['num_frames']
        interval = annotation['interval']

        if self.is_training:
            # random sampling
            if self.random_sampling:
                sample_frames = random.sample(range(total_frames), self.num_frame)
                sample_frames.sort()
            # segment-based sampling
            else:                
                segment_duration = total_frames // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame)
        else:
            # random sampling
            if self.random_sampling:
                sample_frames = random.sample(range(total_frames), self.num_frame)
                sample_frames.sort()
            # segment-based sampling
            else:
                segment_duration = total_frames // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame)

        return [(frame, int(fid * annotation['interval'])) for fid in sample_frames]

    def load_samples(self, frames):
        images, boxes, gt_boxes, actions, activities, members, membership = [], [], [], [], [], [], []
        targets = {}
        fids = []

        for i, (frame, fid) in enumerate(frames):
            vid, cid = frame
            fids.append(fid)
            img = Image.open(self.image_path + '/%s/%s/images/frames_%d.jpg' % (vid, cid, fid))
            image_w, image_h = img.width, img.height
            img = self.transform(img)
            images.append(img)

            num_boxes = self.anns[frame]['boxes'].shape[0]

            for box in self.anns[frame]['boxes']:
                x_c, y_c, w, h = box
                gt_boxes.append([x_c / image_w, y_c / image_h, w / image_w, h / image_h])

            temp_boxes = np.ones((num_boxes, 4))
            for j, track in enumerate(self.tracks[(vid, cid)][fid]):                
                _id, x1, y1, x2, y2 = track

                if x1 < 0.0 and y2 < 0.0:
                    x1, y1, x2, y2 = 0.0, 0.0, 1e-8, 1e-8

                x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1

                if _id <= num_boxes:
                    temp_boxes[int(_id - 1)] = np.array([x_c, y_c, w, h])

            boxes.append(temp_boxes)
            actions = [self.anns[frame]['actions']]
            activities = [self.anns[frame]['activities']]
            members = [self.anns[frame]['members']]
            membership = [self.anns[frame]['membership']]

            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], (self.num_boxes - len(boxes[-1])) * [[0.0, 0.0, 0.0, 0.0]]])

            if len(actions[-1]) != self.num_boxes:
                actions[-1] = torch.cat((actions[-1], torch.tensor((self.num_boxes - len(actions[-1])) * [self.num_class + 1])))

            if members[-1].shape[1] != self.num_boxes:
                members[-1] = torch.hstack(
                    (members[-1], torch.zeros((members[-1].shape[0], self.num_boxes - members[-1].shape[1]))))

            if len(membership) != self.num_boxes:
                membership[-1] = torch.cat((membership[-1], torch.tensor((self.num_boxes - len(membership[-1])) * [-1])))

        images = torch.stack(images)
        boxes = np.vstack(boxes).reshape([self.num_frame, -1, 4])
        gt_boxes = np.vstack(gt_boxes).reshape([self.num_frame, -1, 4])
        actions = torch.stack(actions)
        membership = torch.stack(membership)

        if len(activities) == 0:
            activities = torch.tensor(activities)
            members = torch.tensor(activities)
        else:
            activities = torch.stack(activities)
            members = torch.stack(members)

        boxes = torch.from_numpy(boxes).float()
        gt_boxes = torch.from_numpy(gt_boxes).float()

        targets['actions'] = actions
        targets['activities'] = activities
        targets['boxes'] = boxes
        targets['gt_boxes'] = gt_boxes
        targets['members'] = members
        targets['membership'] = membership

        infos = {'vid': vid, 'sid': cid, 'fid': fids, 'key_frame': self.anns[frame]['key_frame']}

        return images, targets, infos
