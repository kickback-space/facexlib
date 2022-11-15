import argparse
import cv2
import torch

from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_detection


def export_onnx(model, input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.Tensor([input]).to(device).permute(0,3,1,2) # Convert to NCHW
    torch.onnx.export(model,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      'restina_resnet50.onnx',  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=True,
                      input_names=['image'],  # the model's input names
                      output_names=['bboxes', 'classes', 'ldms']  # the model's output names
                      )


def main(args):
    # initialize model
    det_net = init_detection_model(args.model_name, half=args.half)

    img = cv2.imread(args.img_path)
    with torch.no_grad():
        bboxes = det_net.detect_faces(img, 0.97)
        # x0, y0, x1, y1, confidence_score, five points (x, y)
        print (bboxes)
        visualize_detection(img, bboxes, args.save_path)
    export_onnx(det_net, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test.jpg')
    parser.add_argument('--save_path', type=str, default='test_detection.png')
    parser.add_argument(
        '--model_name', type=str, default='retinaface_resnet50', help='retinaface_resnet50 | retinaface_mobile0.25')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args)
