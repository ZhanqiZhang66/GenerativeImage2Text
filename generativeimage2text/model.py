from .torch_common import resize_2d_pos_embed
import torch
from torch import nn
from .layers.CLIP import clip
from .layers.decoder import CaptioningModel
from .layers.decoder import (TransformerDecoderTextualHead,
                             AutoRegressiveBeamSearch, GeneratorWithBeamSearch)


def get_git_model(tokenizer, param):
    visual_feature_size=param.get('visual_feature_size', 768)

    image_encoder = get_image_encoder(
        param.get('image_encoder_type', 'CLIPViT_B_16'),
        input_resolution=param.get('test_crop_size', 224),
    )
    text_decoder = TransformerDecoderTextualHead(
        visual_feature_size=visual_feature_size,
        vocab_size=30522,
        hidden_size=768,
        num_layers=6,
        attention_heads=12,
        feedforward_size=768* 4,
        max_caption_length=1024,
        mask_future_positions=True,
        padding_idx=0,
        decoder_type='bert_en',
        visual_projection_type='linearLn',
    )
    #decoder = AutoRegressiveBeamSearch(
        #eos_index=tokenizer.sep_token_id,
        #max_steps=40,
        #beam_size=1,
        #per_node_beam_size=1,
        #fix_missing_prefix=True,
    #)
    decoder = GeneratorWithBeamSearch(
        eos_index=tokenizer.sep_token_id,
        #max_steps=40,
        # max_steps=1024,
        max_steps=100,
        beam_size=4,
        length_penalty=0.6,
    )
    pose_encoder = nn.Sequential(
        nn.Linear(32, visual_feature_size),
        nn.ReLU(),
        nn.Linear(visual_feature_size, visual_feature_size),
        nn.ReLU(),
        nn.Linear(visual_feature_size, visual_feature_size * 4),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(visual_feature_size * 4, visual_feature_size),
        nn.ReLU(),
    )

    model = CaptioningModel(
        image_encoder,
        text_decoder,
        pose_encoder,
        decoder=decoder,
        sos_index=tokenizer.cls_token_id,
        eos_index=tokenizer.sep_token_id,
        tokenizer=tokenizer,
        use_history_for_infer=True,
        loss_type='smooth',
        num_image_with_embedding=param.get('num_image_with_embedding')
    )
    return model

def get_image_encoder(encoder_type, input_resolution=224):
    name_map = {
        'CLIPViT_B_16': 'ViT-B/16',
        'CLIPViT_L_14': 'ViT-L/14',
    }
    name_in_clip = name_map[encoder_type]
    model, _ = clip.load(name_in_clip, device='cpu', jit=False)
    model = model.train()
    ret = model.visual
    ret.to(torch.float32)
    ret.output_grid = True
    ret.grid_after_ln = True
    if ret.input_resolution != input_resolution:
        if encoder_type in ['CLIPViT_B_16', 'CLIPViT_L_14']:
            pos = ret.positional_embedding
            patch_size = ret.conv1.kernel_size[0]
        else:
            pos = ret.attnpool.positional_embedding
            patch_size = 32
        p2 = resize_2d_pos_embed(pos,
                            ret.input_resolution,
                            patch_size,
                            input_resolution)
        ret.input_resolution = input_resolution
        if encoder_type in ['CLIPViT_B_16', 'CLIPViT_L_14']:
            ret.positional_embedding = torch.nn.Parameter(p2)
        else:
            ret.attnpool.positional_embedding = torch.nn.Parameter(p2)
    return ret

