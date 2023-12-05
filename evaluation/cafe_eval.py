# ------------------------------------------------------------------------
# Modified from JRDB-ACT (https://github.com/JRDB-dataset/jrdb_toolkit/tree/main/Action%26Social_grouping_eval)
# ------------------------------------------------------------------------
from collections import defaultdict, Counter
import numpy as np
import copy


def make_image_key(v_id, c_id, f_id):
  """Returns a unique identifier for a video id & clip id & frame id"""
  return "%d,%d,%d" % (int(v_id), int(c_id), int(f_id))

def make_clip_key(image_key):
  """Returns a unique identifier for a video id & clip id"""
  v_id = image_key.split(',')[0]
  c_id = image_key.split(',')[1]
  return "%d,%d" % (int(v_id), int(c_id))  

def read_text_file(text_file, eval_type, mode):
    """Loads boxes and class labels from a CSV file in the cafe format.

    Args:
      text_file: A file object.
      mode: 'gt' or 'pred'
      eval_type: 
        'gt_base': Eval type for trained model with ground turth actor tracklets as inputs.
        'detect_base': Eval type for trained model with tracker actor tracklets as inputs.

    Returns:
      boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
      g_labels: A dictionary mapping each unique image key (string) to a list of
        integer group id labels, matching the corresponding box in 'boxes'.
      act_labels: A dictionary mapping each unique image key (string) to a list of
        integer group activity class lables, matching the corresponding box in `boxes`.
      a_scores: A dictionary mapping each unique image key (string) to a list of
        actor confidence score values lables, matching the corresponding box in `boxes`.
      g_scores: A dictionary mapping each unique image key (string) to a list of
        group confidence score values lables, matching the corresponding box in `boxes`.
    """
    boxes = defaultdict(list)
    g_labels = defaultdict(list)
    act_labels = defaultdict(list)
    a_scores = defaultdict(list)
    g_scores = defaultdict(list)
    # reads each row in text file.
    with open(text_file.name) as r:
        for line in r.readlines():
            row = line[:-1].split(' ')
            # makes image key.
            image_key = make_image_key(row[0], row[1], row[2])
            # box coordinates.
            x1, y1, x2, y2 = [float(n) for n in row[3:7]]
            # actor confidence score.
            if eval_type == 'detect_base' and mode == 'pred':
                a_score = float(row[10])
            else:
                a_score = 1.0
            # group confidence score.
            if mode == 'gt':
                g_score = None
            elif mode == 'pred':
                g_score = float(row[9])
            # group identity document.
            group_id = int(row[7])
            # group activity label.
            activity = int(row[8])

            boxes[image_key].append([x1, y1, x2, y2])
            g_labels[image_key].append(group_id)
            act_labels[image_key].append(activity)
            a_scores[image_key].append(a_score)
            g_scores[image_key].append(g_score)
    return boxes, g_labels, act_labels, a_scores, g_scores

def actor_matching(pred_boxes, pred_a_scores, gt_boxes):
    """matches prediction tracklets and ground truth tracklets.

    Args:
      pred_boxes: A dictionary mapping each unique image key (string) to a list of
        prediction boxes, given as coordinates [y1, x1, y2, x2]. it has same permutation
        in image keys of each clip key.
      pred_a_scores: A dictionary mapping each unique image key (string) to a list of
        actor confidence score values lables, matching the corresponding box in `pred_boxes`.
      gt_boxes: A dictionary mapping each unique image key (string) to a list of
        ground truth boxes, given as coordinates [y1, x1, y2, x2]. it has same permutation
        in image keys of each clip key.
    
    Returns:
      matching_results: A dictionary mapping each unique clip key (string) to a list of
        id matching results between prediction tracklets and ground truth tracklets, given as 
        {[prediction id: [ground truth id]]}.
    """
    image_keys = pred_boxes.keys()
    frame_list = defaultdict(list)
    matching_results = defaultdict(list)
    for image_key in image_keys:
        clip_key = make_clip_key(image_key)
        frame_list[clip_key].append(image_key)
    clip_keys = frame_list.keys()
    for clip_key in clip_keys:
        matching = defaultdict(list)
        # IoU matrix
        iou_matrix = np.zeros((len(pred_boxes[frame_list[clip_key][0]]), len(gt_boxes[frame_list[clip_key][0]])))
        # confidence score for prediction tracklets.
        confidence_mean = np.zeros(len(pred_boxes[frame_list[clip_key][0]]))
        # image numbers of clip.
        frame_len = len(frame_list[clip_key])
        # puts sum of IoUs on the IoU matrix and sum of confidence score.
        for image_key in frame_list[clip_key]:
            for i, pred_box in enumerate(pred_boxes[image_key]):
                if pred_box[2] != 0 and pred_box[2] != -1:
                    confidence_mean[i] += pred_a_scores[image_key][i]
                    for j, gt_box in enumerate(gt_boxes[image_key]):
                        if gt_box[2] != 0 and gt_box[2] != -1:
                            iou_matrix[i,j] += IoU(pred_box, gt_box)
        # takes each mean of IoU and confidence score.
        confidence_mean = confidence_mean / frame_len
        iou_matrix = iou_matrix / frame_len
        # sorts by confidence score.
        sorted_scores = sorted(confidence_mean, reverse=True)

        # matching algorithm
        duplicated_score = 0
        for score in sorted_scores:
            if duplicated_score == score:
                continue
            else:
                for i in (np.where(confidence_mean == score)[0]):
                    if max(iou_matrix[i]) > 0.5:
                        j = np.where(iou_matrix[i] == max(iou_matrix[i]))[0][0]
                        matching[i].append(j)
                        iou_matrix[:,j] = 0
        
        # matching results
        matching_results[clip_key].append(matching)
    return matching_results          
 

def make_groups(boxes, g_labels, act_labels, g_scores):
    """combines boxes, activity, score to same group, same image
       
    Returns:
      groups_ids: A dictionary mapping each unique clip key (string) to a list of
        actor ids of each 'g_label'.
      groups_activity: A dictionary mapping each unique clip key (string) to a list of
        group activity class labels.
      groups_score: A dictionary mapping each unique clip key (string) to a list of
        group confidence score.
    """
    image_keys = boxes.keys()
    groups_activity = defaultdict(list)
    groups_score = defaultdict(list)
    groups_ids = defaultdict(list)
    frame_list = defaultdict(list)
    # makes clip key.
    for image_key in image_keys:
        clip_key = make_clip_key(image_key)
        frame_list[clip_key].append(image_key)
    clip_keys = frame_list.keys()
    for clip_key in clip_keys:
        group_ids = defaultdict(list)
        group_activity = defaultdict(set)
        group_score = defaultdict(set)
        for i, (g_label, act_label, g_score) in enumerate(
                zip(g_labels[frame_list[clip_key][0]], act_labels[frame_list[clip_key][0]],
                    g_scores[frame_list[clip_key][0]])):
            group_ids[g_label].append(i)
            group_activity[g_label].add(act_label)
            group_score[g_label].add(g_score)
        groups_ids[clip_key].append(group_ids)
        groups_activity[clip_key].append(group_activity)
        groups_score[clip_key].append(group_score)

    return groups_ids, groups_activity, groups_score


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
      labelmap_file: A file object containing a label map protocol buffer.

    Returns:
      labelmap: The label map in the form used by the object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
      class_ids: A set containing all of the valid class id integers.
    """
    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""
    for line in labelmap_file:
        if line.startswith("  name:"):
            name = line.split('"')[1]
        elif line.startswith("  id:") or line.startswith("  label_id:"):
            class_id = int(line.strip().split(" ")[-1])
            labelmap.append({"id": class_id, "name": name})
            class_ids.add(class_id)
    return labelmap, class_ids


def IoU(box1, box2):
    """calculates IoU between two different boxes."""
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter + 1e-8)
    return iou


def cal_group_IoU(pred_group, gt_group):
    """calculates group IoU between two different groups"""
    # Intersection
    Intersection = sum([1 for det_id in pred_group[2] if det_id in gt_group[2]])
  
    # group IoU
    if Intersection != 0:
        group_IoU = Intersection / (len(pred_group[2]) + len(gt_group[2]) - Intersection)
    else:
        group_IoU = 0
    return group_IoU


def calculateAveragePrecision(rec, prec):
    """calculates AP score of each activity class by all-point interploation method."""
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ii = []

    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)

    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def outlier_metric(gt_groups_ids, gt_groups_activity, pred_groups_ids, pred_groups_activity, num_class):
    """calculates Outlier mIoU.

    Args:
      num_class: A number of activity classes.
    
    Returns:
      outlier_mIoU: Mean of outlier IoUs on each clip.      
    """
    clip_IoU = defaultdict(list)
    TP = defaultdict(list)
    clip_keys = pred_groups_ids.keys()
    c_pred_groups_activity = copy.deepcopy(pred_groups_activity)
    c_gt_groups_activity = copy.deepcopy(gt_groups_activity)
    # prediction groups on each class. defines group has members equals or more than two.
    pred_groups = [[clip_key, group_id, pred_groups_ids[clip_key][0][group_id]] for clip_key in clip_keys if
                  clip_key in gt_groups_ids.keys() for group_id in pred_groups_ids[clip_key][0].keys() if
                  c_pred_groups_activity[clip_key][0][group_id].pop() == (num_class + 1)]
    # ground truth groups on each class.
    gt_groups = [[clip_key, group_id, gt_groups_ids[clip_key][0][group_id]] for clip_key in clip_keys if
                 clip_key in gt_groups_ids.keys() for group_id in gt_groups_ids[clip_key][0].keys() if
                 c_gt_groups_activity[clip_key][0][group_id].pop() == (num_class + 1)]
    for clip_key in clip_keys:
        # escapes error that there are not exist pred_image_key on gt.txt.
        if clip_key in gt_groups_ids.keys():
            # groups on same clip
            c_pred_groups = [pred_group for pred_group in pred_groups if pred_group[0] == clip_key]
            c_gt_groups = [gt_group for gt_group in gt_groups if gt_group[0] == clip_key]
            if len(c_pred_groups) != 0 and len(c_gt_groups) != 0:
                # outliers on prediction and ground truth.
                c_pred_ids = [pred_id for c_pred_group in c_pred_groups for pred_id in c_pred_group[2]]
                c_gt_ids = [gt_id for c_gt_group in c_gt_groups for gt_id in c_gt_group[2]]
                # number of True positive outliers.
                TP[clip_key] = sum([1 for pred_id in c_pred_ids if pred_id in c_gt_ids])
                clip_IoU[clip_key] = TP[clip_key] / (len(c_pred_ids) + len(c_gt_ids) - TP[clip_key])
                clip_IoU['total'].append(clip_IoU[clip_key])
            elif len(c_pred_groups) != 0 or len(c_gt_groups) != 0:
                TP[clip_key] = 0
                clip_IoU[clip_key] = 0
                clip_IoU['total'].append(clip_IoU[clip_key])
    # outlier mIoU.
    outlier_mIoU = np.array(clip_IoU['total']).mean()
    return outlier_mIoU * 100.0


def group_mAP_eval(gt_groups_ids, gt_groups_activity, pred_groups_ids, pred_groups_activity, pred_groups_scores,
                   categories, thresh):
    """calculates group mAP.

    Args: 
      categories: A list of group activity classes, given as {name: ,id: }.
      thresh: A group IoU threshold for true positive prediction group condition.

    Returns:
      group_mAP: Mean of group APs on each activity class.
      group_APs; A list of each group AP on each activity class.
    """
    clip_keys = pred_groups_ids.keys()
    # acc on each class.
    group_APs = np.zeros(len(categories))
    for c, clas in enumerate(categories):
        # copy for set funtion to pop.
        c_pred_groups_activity = copy.deepcopy(pred_groups_activity)
        c_gt_groups_activity = copy.deepcopy(gt_groups_activity)
        # prediction groups on each class.
        pred_groups = [
            [clip_key, group_id, pred_groups_ids[clip_key][0][group_id], pred_groups_scores[clip_key][0][group_id]] for
            clip_key in clip_keys if clip_key in gt_groups_ids.keys() for group_id in
            pred_groups_ids[clip_key][0].keys() if
            c_pred_groups_activity[clip_key][0][group_id].pop() == clas['id'] and len(
                pred_groups_ids[clip_key][0][group_id]) >= 2]
        # ground truth groups on each class.
        gt_groups = [[clip_key, group_id, gt_groups_ids[clip_key][0][group_id]] for clip_key in clip_keys if
                     clip_key in gt_groups_ids.keys() for group_id in gt_groups_ids[clip_key][0].keys() if
                     c_gt_groups_activity[clip_key][0][group_id].pop() == clas['id'] and len(
                         gt_groups_ids[clip_key][0][group_id]) >= 2]

        # denominator of Recall.
        npos = len(gt_groups)

        # sorts det_groups in descending order for g_score.
        pred_groups = sorted(pred_groups, key=lambda conf: conf[3], reverse=True)

        TP = np.zeros(len(pred_groups))
        FP = np.zeros(len(pred_groups))

        det = Counter(gt_group[0] for gt_group in gt_groups)

        for key, val in det.items():
            det[key] = np.zeros(val)

        # AP matching algorithm.
        for p, pred_group in enumerate(pred_groups):
            if pred_group[0] in gt_groups_ids.keys():
                gt = [gt_group for gt_group in gt_groups if gt_group[0] == pred_group[0]]
                group_IoU_Max = 0
                for j, gt_group in enumerate(gt):
                    group_IoU = cal_group_IoU(pred_group, gt_group)
                    if group_IoU > group_IoU_Max:
                        group_IoU_Max = group_IoU
                        jmax = j
                # true positive prediction group condition.
                if group_IoU_Max >= thresh:
                    if det[pred_group[0]][jmax] == 0:
                        TP[p] = 1
                        det[pred_group[0]][jmax] = 1
                    else:
                        FP[p] = 1
                else:
                    FP[p] = 1

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        # recall
        rec = acc_TP / npos
        # precision
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
        # group AP on each group activity class
        group_APs[c] = ap * 100    
    # group mAP
    group_mAP = group_APs.mean()
    return group_mAP, group_APs

class GAD_Evaluation():
    def __init__(self, args):
        super(GAD_Evaluation, self).__init__()
        self.eval_type = args.eval_type
        self.categories, self.class_whitelist = read_labelmap(args.labelmap)
        self.gt_boxes, self.gt_g_labels, self.gt_act_labels, _, self.gt_g_scores = read_text_file(args.groundtruth, self.eval_type, mode='gt')
        self.gt_groups_ids, self.gt_groups_activity, _ = make_groups(
            self.gt_boxes, self.gt_g_labels, self.gt_act_labels, self.gt_g_scores)
        

    def evaluate(self, detections):
        pred_boxes, pred_g_labels, pred_act_labels, pred_a_scores, pred_g_scores = read_text_file(detections, self.eval_type, mode='pred')
        pred_groups_ids, pred_groups_activity, pred_groups_scores = make_groups(pred_boxes, pred_g_labels,
                                                                                pred_act_labels,
                                                                                pred_g_scores)        
        group_mAP, group_APs = group_mAP_eval(self.gt_groups_ids, self.gt_groups_activity,
                                              pred_groups_ids, pred_groups_activity, pred_groups_scores,
                                              self.categories, thresh=1.0)
        group_mAP_2, group_APs_2 = group_mAP_eval(self.gt_groups_ids, self.gt_groups_activity,
                                              pred_groups_ids, pred_groups_activity, pred_groups_scores,
                                              self.categories, thresh=0.5)
        outlier_mIoU = outlier_metric(self.gt_groups_ids, self.gt_groups_activity,
                                      pred_groups_ids, pred_groups_activity,
                                      len(self.categories))
        result = {
            'group_APs_1.0': group_APs,            
            'group_mAP_1.0': group_mAP,
            'group_APs_0.5': group_APs_2,
            'group_mAP_0.5': group_mAP_2,
            'outlier_mIoU': outlier_mIoU,
        }
        return result
