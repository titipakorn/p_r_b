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

        self.random_seed = 100

        self.mct_config = dict(
            time_window=4,
            global_match_thresh=0.2,
            bbox_min_aspect_ratio=1.2
        )

        self.sct_config = dict(
            time_window=8,
            continue_time_thresh=2,
            track_clear_thresh=3000,
            match_threshold=0.25,
            merge_thresh=0.15,
            n_clusters=4,
            max_bbox_velocity=0.2,
            detection_occlusion_thresh=0.7,
            track_detection_iou_thresh=0.5,
            process_curr_features_number=0,
            interpolate_time_thresh=10,
            detection_filter_speed=0.6,
            rectify_thresh=0.1
        )

        self.normalizer_config = dict(
            enabled=True,
            clip_limit=.5,
            tile_size=8
        )

        self.visualization_config = dict(
            show_all_detections=True,
            max_window_size=(1920, 1080),
            stack_frames='vertical'
        )

        self.analyzer = dict(
            enable=False,
            show_distances=True,
            save_distances='',
            concatenate_imgs_with_distances=True,
            plot_timeline_freq=0,
            save_timeline='',
            crop_size=(32, 64)
        )

        self.embeddings = dict(
            save_path='',
            # Use it with `analyzer['enable'] = True` to save crops of objects
            use_images=True,
            step=0  # Equal to subdirectory for `save_path`
        )

        self.footfall_config = dict(
            p_in=[{"x": 576, "y": 718}, {"x": 586, "y": 1}, {
                "x": 2, "y": 0}, {"x": 4, "y": 717}, {"x": 575, "y": 718}],
            p_out=[{"x": 584, "y": 719}, {"x": 601, "y": 0}, {
                "x": 1279, "y": 2}, {"x": 1279, "y": 717}, {"x": 585, "y": 719}]
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
