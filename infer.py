import torch
import argparse
import sys
from utils.data_utils import process_input
from utils.vis_utils import save_obj
from model.backbone import MiDasCore
from model.decoder import VisualEncoder, DepthEncoder, ObjectDecoder, PromptEncoder
from model.model import MVSEstimate, MVS2D
from groundingdino.util.inference import load_model, load_image, predict

def parse_args(args):
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--img_path", default=None, type=str)
    parser.add_argument("--out_path", default='output.png', type=str)

    return parser.parse_args(args)

def detect_one_image(image_path, device):
    TEXT_PROMPT = "tree"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    detect_model = load_model('config/GroundingDINO_SwinT_OGC.py', "checkpoints/groundingdino_swint_ogc.pth", device)
    detect_model.eval()
    with torch.no_grad():
        _, image = load_image(image_path)
        boxes, _, _ = predict(
            model=detect_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=device
        )
    del detect_model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return boxes

def main(args):
    args = parse_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ## 定义Model
    ## 视觉backbone
    visual_backbone = MiDasCore(encoder='vitb', keep_aspect_ratio=True, img_size=518, denorm=True, do_resize=True, freeze_bn=True, trainable=False)
    ## 定义深度backbone
    backone = MVS2D(backbone=visual_backbone, max_depth=80.0, is_rel=False, return_depth_feature=True, mono_only=True)
    ## 定义四个解码器
    visual_encoder = VisualEncoder(num_layers=3, in_chans=768, d_model=384, nhead=8, n_points=8, dim_feedforward=384, multi_scale=False)
    depth_encoder = DepthEncoder(num_layers=3, in_chans=128, d_model=384, nhead=2,dim_feedforward=384)
    object_decoder = ObjectDecoder(num_layers=6, d_ffn=256, d_model=384, dim_feedforward=384, n_levels=4, n_points=8, n_heads=6, multi_scale=False)
    prompy_encoder = PromptEncoder(embed_dim=384)

    model = MVSEstimate(is_rel=False, d_model=384, num_queries=50, max_depth=80.0, has_mask=True, has_depth=True, multi_scale=False, backbone=backone, visual_encoder=visual_encoder, depth_encoder=depth_encoder, object_decoder=object_decoder, prompt_encoder=prompy_encoder)

    ## 加载模型
    ckpt_path = 'checkpoints/whole_pipeline.pth'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    device = 'cpu'
    if device == 'cuda':
        model = model.cuda()

    ## 加载数据输入
    ## 图片输入
    if args.img_path is None:
        img_path = 'demo.jpeg'
    else:
        img_path = args.img_path
    boxes = detect_one_image(img_path, device)
    try:
        assert len(boxes.shape) == 2 and boxes.shape[1] == 4
    except:
        print('Detecte tree failed.')
        exit()
    input_dict = process_input(img_path, boxes, device)

    ## 测试
    model.eval()
    with torch.no_grad():
        preds = model(input_dict)
        obj = preds['obj'][input_dict['mask_2d']]
        for i in range(len(obj)):
            print(f'Pred tree {i+1}: DBH: {obj[i][1]:.4f} cm, Height: {obj[i][0] / 100.0:.4f} m')
        save_obj(input_dict['image'], input_dict['position'][input_dict['mask_2d']], obj, args.out_path)
    ## 保存为图像

if __name__ == "__main__":
    main(sys.argv[1:])