import sclip
import clip

def get_clip(name, device="cpu", training=False, use_clip_surgery=True):

    if use_clip_surgery:
        model, _preprocess = sclip.load(name, device=device)
    else:
        model, _preprocess = clip.load(name, device=device)

    for p in model.parameters():
        p.requires_grad_(training)
    model.train(training)

    return model


def get_clip_imgenc(name, device="cpu", training=False, use_clip_surgery=True):
    clipmodel = get_clip(name, device, training, use_clip_surgery=use_clip_surgery)
    return clipmodel.visual


def get_clip_textenc(name, device="cpu", training=False):
    clipmodel = get_clip(name, device, training)
    # delete unused parameters
    del clipmodel.visual
    del clipmodel.logit_scale
    clipmodel.forward = clipmodel.encode_text

    return clipmodel
