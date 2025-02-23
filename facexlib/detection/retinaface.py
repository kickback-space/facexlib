import time
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models._utils import IntermediateLayerGetter as IntermediateLayerGetter

from facexlib.detection.align_trans import (
    get_reference_facial_points,
    warp_and_crop_face,
)
from facexlib.detection.retinaface_net import (
    FPN,
    SSH,
    MobileNetV1,
    make_bbox_head,
    make_class_head,
    make_landmark_head,
)
from facexlib.detection.retinaface_utils import (
    PriorBox,
    batched_decode,
    batched_decode_landm,
    decode,
    decode_landm,
    fast_decode_landm,
    py_cpu_nms,
)
from facexlib.tensorrt.trt_model import TRTModel


def generate_config(network_name):

    cfg_mnet = {
        "name": "mobilenet0.25",
        "min_sizes": [[16, 32], [64, 128], [256, 512]],
        "steps": [8, 16, 32],
        "variance": [0.1, 0.2],
        "clip": False,
        "loc_weight": 2.0,
        "gpu_train": True,
        "batch_size": 32,
        "ngpu": 1,
        "epoch": 250,
        "decay1": 190,
        "decay2": 220,
        "image_size": 640,
        "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
        "in_channel": 32,
        "out_channel": 64,
    }

    cfg_re50 = {
        "name": "Resnet50",
        "min_sizes": [[16, 32], [64, 128], [256, 512]],
        "steps": [8, 16, 32],
        "variance": [0.1, 0.2],
        "clip": False,
        "loc_weight": 2.0,
        "gpu_train": True,
        "batch_size": 24,
        "ngpu": 4,
        "epoch": 100,
        "decay1": 70,
        "decay2": 90,
        "image_size": 840,
        "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
        "in_channel": 256,
        "out_channel": 256,
    }

    if network_name == "mobile0.25":
        return cfg_mnet
    elif network_name == "resnet50":
        return cfg_re50
    else:
        raise NotImplementedError(f"network_name={network_name}")


class RetinaFace(nn.Module):
    def __init__(
        self,
        model_path,
        network_name="resnet50",
        half=False,
        phase="test",
        load_trt=True,
    ):
        super(RetinaFace, self).__init__()
        self.half_inference = half
        cfg = generate_config(network_name)
        self.backbone = cfg["name"]

        self.model_name = f"retinaface_{network_name}"
        self.cfg = cfg
        self.phase = phase
        self.target_size, self.max_size = 1600, 2150
        self.resize, self.scale, self.scale1 = 1.0, None, None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_tensor = torch.tensor([[[[104.0]], [[117.0]], [[123.0]]]]).to(
            self.device
        )

        self.reference = get_reference_facial_points(default_square=True)
        # Build network.
        backbone = None
        if cfg["name"] == "mobilenet0.25":
            backbone = MobileNetV1()
            self.body = IntermediateLayerGetter(backbone, cfg["return_layers"])
        elif cfg["name"] == "Resnet50":
            import torchvision.models as models

            backbone = models.resnet50(pretrained=False)
            self.body = IntermediateLayerGetter(backbone, cfg["return_layers"])

        in_channels_stage2 = cfg["in_channel"]
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]

        out_channels = cfg["out_channel"]
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = make_class_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.BboxHead = make_bbox_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.LandmarkHead = make_landmark_head(fpn_num=3, inchannels=cfg["out_channel"])

        self.trt_retina = TRTModel(model_path, device=0) if load_trt else None

        self.priorbox = PriorBox(self.cfg, image_size=(120, 160))
        self.priors = self.priorbox.forward().to(self.device)

    def forward(self, inputs):
        priors = self.priorbox.forward().to(self.device)
        inputs = inputs - self.mean_tensor
        out = self.body(inputs)

        if self.backbone == "mobilenet0.25" or self.backbone == "Resnet50":
            out = list(out.values())
        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat(
            [self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        classifications = torch.cat(
            [self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        tmp = [self.LandmarkHead[i](feature) for i, feature in enumerate(features)]
        ldm_regressions = torch.cat(tmp, dim=1)

        bbox_regressions = bbox_regressions[0]
        bbox_regressions = torch.cat(
            (
                priors[:, :2]
                + bbox_regressions[:, :2] * self.cfg["variance"][0] * priors[:, 2:],
                priors[:, 2:]
                * torch.exp(bbox_regressions[:, 2:] * self.cfg["variance"][1]),
            ),
            1,
        )
        bbox_regressions[:, :2] -= bbox_regressions[:, 2:] / 2
        bbox_regressions[:, 2:] += bbox_regressions[:, :2]

        ldm_regressions = ldm_regressions[0]

        left = priors[:, :2]
        right = priors[:, 2:]
        landms = torch.zeros_like(ldm_regressions)

        landms[:, :2] = left + ldm_regressions[:, :2] * self.cfg["variance"][0] * right
        landms[:, 2:4] = (
            left + ldm_regressions[:, 2:4] * self.cfg["variance"][0] * right
        )
        landms[:, 4:6] = (
            left + ldm_regressions[:, 4:6] * self.cfg["variance"][0] * right
        )
        landms[:, 6:8] = (
            left + ldm_regressions[:, 6:8] * self.cfg["variance"][0] * right
        )
        landms[:, 8:10] = (
            left + ldm_regressions[:, 8:10] * self.cfg["variance"][0] * right
        )

        bbox_regressions = bbox_regressions.unsqueeze(0)
        landms = landms.unsqueeze(0)
        if self.phase == "train":
            output = (bbox_regressions, classifications, landms, priors)
        else:
            output = (
                bbox_regressions,
                F.softmax(classifications, dim=-1),
                landms,
                priors,
            )
        return output

    def __detect_faces(self, inputs):
        # get scale
        height, width = inputs.shape[2:]
        self.scale = np.array([width, height, width, height], dtype=np.float32)
        tmp = [
            width,
            height,
            width,
            height,
            width,
            height,
            width,
            height,
            width,
            height,
        ]
        self.scale1 = np.array(tmp, dtype=np.float32)
        # forward
        loc, landmarks, conf, priors = self.trt_retina.infer(inputs)
        return loc, conf, landmarks, priors

    # single image detection
    def transform(self, image, use_origin_size):
        # convert to opencv format
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = image.astype(np.float32)

        # testing scale
        im_size_min = min(image.shape[0:2])
        im_size_max = max(image.shape[0:2])
        resize = float(self.target_size) / float(im_size_min)

        # prevent bigger axis from being more than max_size
        if round(resize * im_size_max) > self.max_size:
            resize = float(self.max_size) / float(im_size_max)
        resize = 1 if use_origin_size else resize

        # resize
        if resize != 1:
            image = cv2.resize(
                image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR
            )

        # convert to torch.tensor format
        image = image.transpose(2, 0, 1)[None]

        return image, resize

    def detect_faces(
        self,
        image,
        conf_threshold=0.8,
        nms_threshold=0.4,
        use_origin_size=True,
    ):
        image, self.resize = self.transform(image, use_origin_size)
        if self.half_inference:
            image = image.half()
        time.time()
        boxes, conf, landmarks, priors = self.__detect_faces(image)
        boxes = boxes[0]
        conf = conf[0]
        landmarks = landmarks[0]
        boxes = boxes * self.scale / self.resize
        scores = conf[:, 1]
        landmarks = landmarks * self.scale1 / self.resize
        # ignore low scores
        inds = np.where(scores > conf_threshold)[0]
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]
        # sort
        order = scores.argsort()[::-1]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]
        # do NMS
        bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        keep = py_cpu_nms(bounding_boxes, nms_threshold)
        bounding_boxes, landmarks = bounding_boxes[keep, :], landmarks[keep]
        return np.concatenate((bounding_boxes, landmarks), axis=1)

    def __align_multi(self, image, boxes, landmarks, limit=None):

        if len(boxes) < 1:
            return [], []

        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]

        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[2 * j], landmark[2 * j + 1]] for j in range(5)]

            warped_face = warp_and_crop_face(
                np.array(image), facial5points, self.reference, crop_size=(112, 112)
            )
            faces.append(warped_face)

        return np.concatenate((boxes, landmarks), axis=1), faces

    def align_multi(self, img, conf_threshold=0.8, limit=None):

        rlt = self.detect_faces(img, conf_threshold=conf_threshold)
        boxes, landmarks = rlt[:, 0:5], rlt[:, 5:]

        return self.__align_multi(img, boxes, landmarks, limit)

    # batched detection
    def batched_transform(self, frames, use_origin_size):
        """
        Arguments:
            frames: a list of PIL.Image, or torch.Tensor(shape=[n, h, w, c],
                type=np.float32, BGR format).
            use_origin_size: whether to use origin size.
        """
        from_PIL = True if isinstance(frames[0], Image.Image) else False

        # convert to opencv format
        if from_PIL:
            frames = [
                cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR) for frame in frames
            ]
            frames = np.asarray(frames, dtype=np.float32)

        # testing scale
        im_size_min = np.min(frames[0].shape[0:2])
        im_size_max = np.max(frames[0].shape[0:2])
        resize = float(self.target_size) / float(im_size_min)

        # prevent bigger axis from being more than max_size
        if np.round(resize * im_size_max) > self.max_size:
            resize = float(self.max_size) / float(im_size_max)
        resize = 1 if use_origin_size else resize

        # resize
        if resize != 1:
            if not from_PIL:
                frames = F.interpolate(frames, scale_factor=resize)
            else:
                frames = [
                    cv2.resize(
                        frame,
                        None,
                        None,
                        fx=resize,
                        fy=resize,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for frame in frames
                ]

        # convert to torch.tensor format
        if not from_PIL:
            frames = frames.transpose(1, 2).transpose(1, 3).contiguous()
        else:
            frames = frames.transpose((0, 3, 1, 2))
            frames = torch.from_numpy(frames)

        return frames, resize

    def batched_detect_faces(
        self, frames, conf_threshold=0.8, nms_threshold=0.4, use_origin_size=True
    ):
        """
        Arguments:
            frames: a list of PIL.Image, or np.array(shape=[n, h, w, c],
                type=np.uint8, BGR format).
            conf_threshold: confidence threshold.
            nms_threshold: nms threshold.
            use_origin_size: whether to use origin size.
        Returns:
            final_bounding_boxes: list of np.array ([n_boxes, 5],
                type=np.float32).
            final_landmarks: list of np.array ([n_boxes, 10], type=np.float32).
        """
        # self.t['forward_pass'].tic()
        frames, self.resize = self.batched_transform(frames, use_origin_size)
        frames = frames.to(self.device)
        frames = frames - self.mean_tensor

        b_loc, b_conf, b_landmarks, priors = self.__detect_faces(frames)

        final_bounding_boxes, final_landmarks = [], []

        # decode
        priors = priors.unsqueeze(0)
        b_loc = (
            batched_decode(b_loc, priors, self.cfg["variance"])
            * self.scale
            / self.resize
        )
        b_landmarks = (
            batched_decode_landm(b_landmarks, priors, self.cfg["variance"])
            * self.scale1
            / self.resize
        )
        b_conf = b_conf[:, :, 1]

        # index for selection
        b_indice = b_conf > conf_threshold

        # concat
        b_loc_and_conf = torch.cat((b_loc, b_conf.unsqueeze(-1)), dim=2).float()

        for pred, landm, inds in zip(b_loc_and_conf, b_landmarks, b_indice):

            # ignore low scores
            pred, landm = pred[inds, :], landm[inds, :]
            if pred.shape[0] == 0:
                final_bounding_boxes.append(np.array([], dtype=np.float32))
                final_landmarks.append(np.array([], dtype=np.float32))
                continue

            # sort
            # order = score.argsort(descending=True)
            # box, landm, score = box[order], landm[order], score[order]

            # to CPU
            bounding_boxes, landm = pred.cpu().numpy(), landm.cpu().numpy()

            # NMS
            keep = py_cpu_nms(bounding_boxes, nms_threshold)
            bounding_boxes, landmarks = bounding_boxes[keep, :], landm[keep]

            # append
            final_bounding_boxes.append(bounding_boxes)
            final_landmarks.append(landmarks)
        # self.t['forward_pass'].toc(average=True)
        # self.batch_time += self.t['forward_pass'].diff
        # self.total_frame += len(frames)
        # print(self.batch_time / self.total_frame)

        return final_bounding_boxes, final_landmarks
