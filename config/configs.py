from .default import DefaultConfig


# class Config(DefaultConfig):
#     """
#     mAP 85.8, Rank1 94.1, @epoch 175
#     """
#     def __init__(self):
#         super(Config, self).__init__()
#         self.CFG_NAME = 'baseline'
#         self.DATA_DIR = '/nfs/public/datasets/person_reid/Market-1501-v15.09.15'
#         self.PRETRAIN_CHOICE = 'imagenet'
#         self.PRETRAIN_PATH = '/nfs/public/pretrained_models/resnet50-19c8e357.pth'
#
#         self.LOSS_TYPE = 'triplet+softmax+center'
#         self.TEST_WEIGHT = './output/resnet50_175.pth'
#         self.FLIP_FEATS = 'on'


class Config(DefaultConfig):
    """
    mAP 86.2, Rank1 94.4, @epoch 185
    """

    def __init__(self):
        super(Config, self).__init__()
        self.CFG_NAME = 'baseline'
        self.DATA_DIR = '/nfs/public/datasets/person_reid/Market-1501-v15.09.15'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './models/resnet50-19c8e357.pth'

        self.LOSS_TYPE = 'triplet+softmax+center'
        self.TEST_WEIGHT = './models/resnet50_person_reid_128x64.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = True

        self.sct_config = dict(
            time_window=30,
            continue_time_thresh=60,
            track_clear_thresh=300,
            match_threshold=0.475,
            merge_thresh=0.3,
            n_clusters=4,
            max_bbox_velocity=0.2,
            detection_occlusion_thresh=0.7,
            track_detection_iou_thresh=0.3
        )

        self.footfall_config = dict(
            p_in=[{"x": 576, "y": 718}, {"x": 586, "y": 1}, {
                "x": 2, "y": 0}, {"x": 4, "y": 717}, {"x": 575, "y": 718}],
            p_out=[{"x": 584, "y": 716}, {"x": 599, "y": 2}, {
                "x": 1116, "y": 3}, {"x": 1028, "y": 716}, {"x": 585, "y": 715}],
        )


# class Config(DefaultConfig):
#     def __init__(self):
#         super(Config, self).__init__()
#         self.CFG_NAME = 'baseline'
#         self.DATA_DIR = '/nfs/public/datasets/person_reid/Market-1501-v15.09.15'
#         self.PRETRAIN_CHOICE = 'imagenet'
#         self.PRETRAIN_PATH = '/nfs/public/pretrained_models/resnet50-19c8e357.pth'
#         self.COS_LAYER = True
#         self.LOSS_TYPE = 'softmax'
#         self.TEST_WEIGHT = './output/resnet50_185.pth'
#         self.FLIP_FEATS = 'off'
#         self.RERANKING = True
