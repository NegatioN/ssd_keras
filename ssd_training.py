# coding=utf-8
import tensorflow as tf
import keras.backend as K

class MultiboxLoss(object):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes + 1
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard
        self.match_threshold = 0.5

    def _l1_smooth_loss(self, y_true, y_pred):
        """Compute L1-smooth loss.
        # Arguments
            y_true: Ground truth bounding boxes,
                tensor of shape (?, num_boxes, 4).
            y_pred: Predicted bounding boxes,
                tensor of shape (?, num_boxes, 4).
        # Returns
            l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).
        # References
            https://arxiv.org/abs/1504.08083
        """
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)


    def _softmax_loss(self, y_true, y_pred):
        """Compute softmax loss.
        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, num_classes).
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, num_classes).
        # Returns
            softmax_loss: Softmax loss, tensor of shape (?, num_boxes).
        """
        y_pred = K.maximum(K.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = - K.sum(y_true * K.log(y_pred), axis=-1)
        return softmax_loss

    #TODO  Forenkle hele loss-funksjonen. Gjør den lesbar.
    def compute_loss(self, y_true, y_pred):
        """Compute mutlibox loss.
        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                y_true[:, :, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                y_true[:, :, -7:] are all 0.
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, 4 + num_classes + 8).
        # Returns
            loss: Loss for prediction, tensor of shape (?,).
        """
        batch_size = K.shape(y_true)[0]
        num_boxes = K.cast(K.shape(y_true)[1], 'float')
        confs_start = 4 + self.background_label_id
        confs_end = confs_start + self.num_classes
        # loss for all priors
        conf_loss = self._softmax_loss(y_true[:, :, confs_start:confs_end],
                                       y_pred[:, :, confs_start:confs_end])
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # get positives loss
        num_pos = K.sum(y_true[:, :, -8], axis=-1)
        pos_loc_loss = K.sum(loc_loss * y_true[:, :, -8], axis=1)
        pos_conf_loss = K.sum(conf_loss * y_true[:, :, -8], axis=1)

        # get negatives loss, we penalize only confidence here
        num_neg = K.minimum(self.neg_pos_ratio * num_pos, num_boxes - num_pos)
        pos_num_neg_mask = K.greater(num_neg, 0)
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat(values=[num_neg, [(1 - has_min) * self.negatives_for_hard]], axis=0)

        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_neg, tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)
        neg_conf_loss = self.find_negative_confidence(y_true, y_pred, conf_loss, num_neg_batch, num_boxes, batch_size)

        # loss is sum of positives and negatives
        total_loss = pos_conf_loss + neg_conf_loss
        total_loss += (self.alpha * pos_loc_loss) / num_pos
        total_loss /= (num_pos + tf.to_float(num_neg_batch))
        return total_loss

    def find_negative_confidence(self, y_true, y_pred, conf_loss, num_neg_batch, num_boxes, batch_size):
        confs_start = 4 + self.background_label_id +1
        confs_end = confs_start + self.num_classes -1
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end], axis=2)
        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]),
                                 k=num_neg_batch)
        batch_idx = find_batch_indexes(num_neg_batch, batch_size)
        full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) + tf.reshape(indices, [-1]))
        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]),
                                  full_indices)
        neg_conf_loss = tf.reshape(neg_conf_loss,
                                   [batch_size, num_neg_batch])
        return K.sum(neg_conf_loss, axis=1)


def find_batch_indexes(num_neg_batch, batch_size):
    batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
    return tf.tile(batch_idx, (1, num_neg_batch))