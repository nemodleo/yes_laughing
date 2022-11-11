from models import get_stn #, ResnetClassifier
from utils.download import find_model, PRETRAINED_TEST_HYPERPARAMS

def load_stn(args, load_classifier=False, device='cuda'):
    # try:
    #     supersize = args.crop_size
    # except:
    #     supersize = args.real_size
    supersize = args.real_size #
    ckpt, using_our_pretrained_model = find_model(args.ckpt)

    if using_our_pretrained_model and not args.override:  # Load our test time hyperparameters automatically
        hyperparameters = PRETRAINED_TEST_HYPERPARAMS[args.ckpt]
        for hyper_name, hyper_val in hyperparameters.items():
            setattr(args, hyper_name, hyper_val)
    t_ema = get_stn(args.transform, flow_size=args.flow_size, supersize=supersize,
                    channel_multiplier=args.stn_channel_multiplier, num_heads=args.num_heads).to(device)
    t_ema.load_state_dict(ckpt['t_ema'])
    t_ema.eval()  # The STN doesn't use eval-specific ops, so this shouldn't do anything
    return t_ema

    # if load_classifier:  # Also return the cluster classifier if it exists:
    #     if 'classifier' in ckpt:
    #         classifier = ResnetClassifier(args.flow_size, channel_multiplier=args.stn_channel_multiplier,
    #                                       num_heads=2 * args.num_heads, supersize=supersize).to(device)
    #         classifier.load_state_dict(ckpt['classifier'])
    #         classifier.eval()  # Shouldn't do anything
    #         return t_ema, classifier
    #     else:
    #         return t_ema, None
    # else:
    #     return t_ema
