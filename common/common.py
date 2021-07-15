from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of HyMOS')

    parser.add_argument('--dataset', help='Dataset', default='OfficeHome',
                        choices=['OfficeHome', 'DomainNet', 'Office31'], type=str)
    parser.add_argument('--test_domain', help="Domain (or name of file referring to data) for testing", type=str)

    parser.add_argument("--local_rank", type=int,default=0, help='Local rank for distributed learning')
    parser.add_argument('--load_path', help='Path to the loading checkpoint for eval', default=None, type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts', action='store_true')
    parser.add_argument('--pretrain', help="Path to pretrained network", default = 'pretrained/resnet50_SupCLR.pth')

    ##### Training Configurations #####
    parser.add_argument('--lr_init', help='Initial learning rate', default=0.005, type=float)
    parser.add_argument('--weight_decay', help='Weight decay', default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size for style batches',default=32, type=int)

    ##### Objective Configurations #####
    parser.add_argument('--temperature', help='Temperature for similarity', default=0.07, type=float)

    parser.add_argument("--suffix", default="", type=str, help="suffix for log dir")

    #### Style transfer options ####
    parser.add_argument("--adain_probability", type=float, default=0.5, 
            help="Probability to apply adain to each batch image")
    parser.add_argument("--adain_alpha", type=float, default=1.0,
            help="Alpha coefficient for adain style transfer")
    parser.add_argument("--adain_ckpt", type=str, default=None, 
            help="Path to adain checkpoint")

    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()
