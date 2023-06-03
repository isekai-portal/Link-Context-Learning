import argparse

from mmengine.config import Config, DictAction


def prepare_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')

    args, unknown_args = parser.parse_known_args(args)
    if unknown_args:
        raise ValueError(f"Some specified arguments are not used: {unknown_args}")
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    return cfg


if __name__ == '__main__':
    cfg = prepare_args([r'../config/_base_/dataset/mix_pretrain_prob.py'])
    print(cfg.pretty_text)
