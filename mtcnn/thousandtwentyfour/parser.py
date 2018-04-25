import argparse


def parse_args():
    """
    Parse Arguments for MTCNN.

    Returns:
    -------
    * `args`: [argparse object]
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='MTCNN')
    parser.add_argument('--data_dir', type=str, default='/home/ygx/data/deidentified',
                        help='Root directory for the data')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training [default=16]')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='Number of epochs to be run [default 50]')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate [default: 0.01]')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum [default: 0.5]')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--half_precision', type=bool, default=False,
                        help='Whether to train with half precision [default: False]')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Interval at which to log progress. [default: 10]')
    parser.add_argument('--results_dir', type=str, 
                        help='Path to save hyperparameter optimization results')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed [default: 1]')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model fp16 mode.')
    parser.add_argument('--stored_results', type=str, default='',
                        help='Location of hyperparameter stored results')
    parser.add_argument('--num_result', type=int, default=0,
                        help='Which set of hyperparmaters to use for profiling.')
    args = parser.parse_args()
    return args
