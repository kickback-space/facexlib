import argparse
import cv2
import torch

from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_onnx(model, input):
    input, r = model.transform(input, use_origin_size=True)
    input = torch.Tensor(input).to(device)  # Convert to NCHW
    model.eval()
    model.to(device)
    out = model(input)
    for o in out:
        print(o.shape)
    print(out[0], out[1])
    torch.onnx.export(
        model,  # model being run
        input,  # model input (or a tuple for multiple inputs)
        "retina_resnet50.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        verbose=True,
        input_names=["image"],  # the model's input names
        output_names=[
            "bboxes",
            "classes",
            "ldms",
            "priors",
        ],  # the model's output names
    )


def main(args):
    # initialize model
    det_net = init_detection_model(args.model_name, half=args.half, load_trt=False)
    det_net.eval()
    det_net.to(device)
    img = cv2.imread(args.img_path)
    img = cv2.resize(img, (160, 120))
    #    img, r = det_net.transform(img, use_origin_size=True)
    print(img.shape)
    with torch.no_grad():
        bboxes = det_net.detect_faces(img, 0.97)
        # x0, y0, x1, y1, confidence_score, five points (x, y)
        print(bboxes)
        print(bboxes.shape)
    #        visualize_detection(img, bboxes, args.save_path)
    export_onnx(det_net, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="assets/test.jpg")
    parser.add_argument("--save_path", type=str, default="test_detection.png")
    parser.add_argument(
        "--model_name",
        type=str,
        default="retinaface_resnet50",
        help="retinaface_resnet50 | retinaface_mobile0.25",
    )
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    main(args)
