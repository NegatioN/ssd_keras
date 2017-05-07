import matplotlib.pyplot as plt
import numpy as np
import PIL
import copy

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)


def extract_box(box):
    return box[:, 0], box[:, 1], box[:, 2], box[:, 3], box[:, 4], box[:, 5]


def extract_tops(det_conf, det_xmax, det_xmin, det_ymax, det_ymin, top_indices):
    top_conf = det_conf[top_indices]
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    return top_conf, top_xmax, top_xmin, top_ymax, top_ymin


def scale_coords_to_image(img, top_xmax, top_xmin, top_ymax, top_ymin):
    xmin = int(round(top_xmin * img.shape[1]))
    ymin = int(round(top_ymin * img.shape[0]))
    xmax = int(round(top_xmax * img.shape[1]))
    ymax = int(round(top_ymax * img.shape[0]))
    return xmax, xmin, ymax, ymin


def save_boxes(img, box, label_classes, top_indices, top_label_indices, must_be_in_classes):
    _, det_conf, det_xmin, det_ymin, det_xmax, det_ymax = extract_box(box)
    top_conf, top_xmax, top_xmin, top_ymax, top_ymin = extract_tops(det_conf, det_xmax, det_xmin, det_ymax, det_ymin,
                                                                    top_indices)
    annotations = []
    for i in range(top_conf.shape[0]):
        label = int(top_label_indices[i])
        label_name = label_classes[label - 1]
        if label_name not in must_be_in_classes:
            continue
        xmax, xmin, ymax, ymin = scale_coords_to_image(img, top_xmax[i], top_xmin[i], top_ymax[i], top_ymin[i])
        score = top_conf[i]
        annotations.append({"class": label_name, "y": ymin, "x": xmin,
                            "height": ymax - ymin + 1, "width": xmax - xmin + 1, "score": score})

    return annotations


def render_boxes(img, box, label_classes, top_indices, top_label_indices, must_be_in_classes):
    _, det_conf, det_xmin, det_ymin, det_xmax, det_ymax = extract_box(box)
    top_conf, top_xmax, top_xmin, top_ymax, top_ymin = extract_tops(det_conf, det_xmax, det_xmin, det_ymax, det_ymin, top_indices)

    colors = plt.cm.hsv(np.linspace(0, 1, len(label_classes) + 1)).tolist()
    #img = img[:, :, ::-1] + vgg_mean
    plt.imshow(img / 255.)
    currentAxis = plt.gca()
    for i in range(top_conf.shape[0]):
        xmax, xmin, ymax, ymin = scale_coords_to_image(img, top_xmax[i], top_xmin[i], top_ymax[i], top_ymin[i])
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = label_classes[label - 1]
        if label_name not in must_be_in_classes:
            continue
        display_txt = '{:0.2f}, {}, {}'.format(score, label_name, i)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    plt.show()

def wrap_entry(fname, box, top_indices, top_label_indices):
    return {"filename": fname, "box": box, "top_indices": top_indices, "top_label_indices": top_label_indices}

def unwrap_entry(entry):
    img = np.asarray(PIL.Image.open(entry['filename']))
    box = entry['box']
    top_indices = entry['top_indices']
    top_label_indices = entry['top_label_indices']
    return box, img, top_indices, top_label_indices

def get_top_confidence(filenames, boxes, confidence_threshold):
    top_conf = []
    for i, fname in enumerate(filenames):
        # Parse the outputs.
        box = boxes[i]
        det_label, det_conf, _, _, _, _ = extract_box(box)

        # Get detections with confidence higher than what user selected
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= confidence_threshold]

        top_label_indices = det_label[top_indices].tolist()
        top_conf.append(wrap_entry(fname, box, top_indices, top_label_indices))

    return top_conf


def get_top_n(filenames, boxes, num_per_class):
    top_n = []
    for i, fname in enumerate(filenames):
        # Parse the outputs.
        box = boxes[i]
        det_label, _, _, _, _, _ = extract_box(box)

        # Get detections of top N boxes per class
        all_labels = det_label.tolist()

        top_all_labels = []
        for label_index in np.unique(all_labels):
            indexes_of_label = np.where(all_labels == label_index)[0][:num_per_class]
            top_all_labels.append(indexes_of_label)

        top_indices = [item for sublist in top_all_labels for item in sublist]

        top_label_indices = det_label[top_indices].tolist()
        top_n.append(wrap_entry(fname, box, top_indices, top_label_indices))
    return top_n

'''
    Arguments:
        discrim_func: A function that takes in filenames, box-results from model, and a variable to discriminate on
            current impls: get_top_n, get_top_confidence
'''
def save_bboxes(discrim_func, filenames, boxes, discriminator_variable, label_classes, must_be_in_classes=None):
    if not must_be_in_classes:
        must_be_in_classes = label_classes
    top_by_discriminator = discrim_func(filenames, boxes, discriminator_variable)
    img_annos = []
    for entry in top_by_discriminator:
        box, img, top_indices, top_label_indices = unwrap_entry(entry)
        annos = save_boxes(img, box, label_classes, top_indices, top_label_indices, must_be_in_classes)
        img_annos.append({"class": "image", "filename": entry['filename'], "annotations": annos})
    return img_annos

'''
    Arguments:
        discrim_func: A function that takes in filenames, box-results from model, and a variable to discriminate on
            current impls: get_top_n, get_top_confidence
'''
def render(discrim_func, filenames, boxes, discriminator_variable, label_classes, must_be_in_classes=None):
    if not must_be_in_classes:
        must_be_in_classes = label_classes
    top_by_discriminator = discrim_func(filenames, boxes, discriminator_variable)
    for entry in top_by_discriminator:
        box, img, top_indices, top_label_indices = unwrap_entry(entry)
        render_boxes(img, box, label_classes, top_indices, top_label_indices, must_be_in_classes)


def save_top_n(filenames, boxes, num_per_class, label_classes, must_be_in_classes=None):
    return save_bboxes(get_top_n, filenames, boxes, num_per_class, label_classes, must_be_in_classes)


def save_above_threshold(filenames, boxes, confidence_threshold, label_classes, must_be_in_classes=None):
    return save_bboxes(get_top_confidence, filenames, boxes, confidence_threshold, label_classes, must_be_in_classes)


def render_top_n(filenames, boxes, num_per_class, label_classes, must_be_in_classes=None):
    render(get_top_n, filenames, boxes, num_per_class, label_classes, must_be_in_classes)


def render_above_threshold(filenames, boxes, confidence_threshold, label_classes, must_be_in_classes=None):
    render(get_top_confidence, filenames, boxes, confidence_threshold, label_classes, must_be_in_classes)


def render_sloth(img_anno, classes, colors=None):
    if not colors:
        colors = plt.cm.hsv(np.linspace(0, 1, len(classes) + 1)).tolist()
    img = np.asarray(PIL.Image.open(img_anno['filename']))
    plt.imshow(img / 255.)
    currentAxis = plt.gca()
    for i, box in enumerate(img_anno['annotations']):
        display_txt = '{:0.2f}, {}, {}'.format(box['score'], box['class'], i)
        x, y = box["x"], box["y"]
        coords = (x, y), box["width"], box["height"]
        color = colors[classes.index(box['class'])]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(x, y, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    plt.show()

def render_sloth_annotations(img_annotations, classes):
    colors = plt.cm.hsv(np.linspace(0, 1, len(classes) + 1)).tolist()
    for img_anno in img_annotations:
        render_sloth(img_anno, classes, colors)


