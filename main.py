import argparse
import os
from baselines import Prediction

def main(opts):
    # check settings
    print(f"GPU_ID: {opts.gpu_id}")
    if opts.baseline == True:
        framework = Prediction(vars(opts))
        framework.proceed()
    return framework

if __name__ == '__main__':
    node_dict = {"korea_covid": 16, "nyc": 5, "nyc_covid": 5, "busan": 60, "daegu": 61, "seoul":128}
    fe_kernel_size_dict = {"korea_covid": 3, "nyc": 1, "nyc_covid": 1, "busan": 3, "daegu": 1, "seoul":3}
    output_padding_dict = {"korea_covid": 1, "nyc": 0, "nyc_covid": 0, "busan": 1, "daegu": 0, "seoul":1}
    cycle_dict = {"korea_covid": 1, "nyc": 24, "nyc_covid": 1, "busan": 24, "daegu": 24, "seoul":24}
    node_dim_dict = {"korea_covid": 1, "nyc": 1, "nyc_covid": 1, "busan": 2, "daegu": 2, "seoul":2}
    
    # ResMet1D
    n_block_dict = {"korea_covid": 48, "nyc": 48, "nyc_covid": 48, "busan": 48, "daegu": 48, "seoul":12}
    downsample_dict = {"korea_covid": 48, "nyc": 48, "nyc_covid": 48, "busan": 48, "daegu": 48, "seoul":12}
    res_kernel_size_dict = {"korea_covid": 16, "nyc": 16, "nyc_covid": 16, "busan": 16, "daegu": 16, "seoul":8}
    res_stride_dict = {"korea_covid": 24, "nyc": 24, "nyc_covid": 24, "busan": 24, "daegu": 24, "seoul":2}
    od_hidden_dim_dict = {"korea_covid": 64, "nyc": 64, "nyc_covid": 64, "busan": 64, "daegu": 64, "seoul":16}

    parser = argparse.ArgumentParser(description='Settings for Bi-Modal Learning for NETS & OD')

    # Hardware settings
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu_ids: e.g. 0, 1, 2, 3, 4, 5, 6, 7')

    # Common settings
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')

    # ResNet1D model settings
    parser.add_argument('--base_filters', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=16)
    parser.add_argument('--res_stride', type=int, default=24)
    parser.add_argument('--groups', type=int, default=32)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--n_block', type=int, default=48)
    parser.add_argument('--downsample_gap', type=int, default=6)
    parser.add_argument('--increasefilter_gap', type=int, default=12)
    parser.add_argument('--use_do', action='store_true', default=True)

    # Linear model settings
    parser.add_argument('--linear_kernel_size', type=int, default=25)
    parser.add_argument('--enc_in', type=int, default=8)
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='busan')
    parser.add_argument('--seq_day', type=int, default=14)
    parser.add_argument('--pred_day', type=int, default=7)
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Rest for validation')    
    parser.add_argument('--test_ratio', type=float, default=0.25, help='For dataset Train/Test Split')
    parser.add_argument('--time_snapshot', type=int, default=365, help='365 Days')    
    parser.add_argument('--num_node', type=int, default=10)
    parser.add_argument('--cycle', type=int, default=24)
    parser.add_argument('--node_feature_dim', type=int, default=2)
    parser.add_argument('--khop', type=int, default=0)

    # For baselines
    parser.add_argument('--baseline', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='DLinear')
    
    # [PatchTST, Autoformer]
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--fc_dropout', type=float, default=0.2)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--patch_len', type=int, default=4)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--padding_patch', type=str, default='end')
    parser.add_argument('--revin', type=int, default=0)
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--patch_kernel_size', type=int, default=25)
    parser.add_argument('--use_amp', action='store_false', default=False)

    # [TimesNet]
    parser.add_argument('--top_k', type=int, default=5)

    # Multi-GPU
    parser.add_argument('--multi_gpu', type=int, default=0)
    
    opts = parser.parse_args()
    opts.num_node = node_dict[opts.dataset]
    opts.fe_kernel_size = fe_kernel_size_dict[opts.dataset]
    opts.cycle = cycle_dict[opts.dataset]
    opts.output_padding = output_padding_dict[opts.dataset]
    opts.node_feature_dim = node_dim_dict[opts.dataset]
    opts.n_block = n_block_dict[opts.dataset]
    opts.downsample_gap = downsample_dict[opts.dataset]
    opts.kernel_size = res_kernel_size_dict[opts.dataset]
    opts.res_stride = res_stride_dict[opts.dataset]
    opts.od_hidden_dim = od_hidden_dim_dict[opts.dataset]

    # set gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ['CUDA_VISIBLE_DEVICES']= opts.gpu_id

    args = vars(opts)
    print('############ Arguments ############')
    print(args)
    
    print('############ Print Option Items ############')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('############################################')
    main(opts)