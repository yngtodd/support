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
<<<<<<< HEAD:mtcnn/twofiftysix/parser.py
    parser.add_argument('--data_dir', type=str, 
                        default='/gpfs/alpinetds/proj-shared/csc276/yngtodd/data/test_data',
=======
    parser.add_argument('--data_dir', type=str, default='/home/ygx/data/shortsynth',
>>>>>>> a7172221c60506840e51cd156b9145a0eeafba4a:mtcnn/parser.py
                        help='Root directory for the data')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training [default=16]')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to be run [default=50]')
    parser.add_argument('--optimizer', type=int, default=0,
                        help='Choice of optimizer [default=0 {Adam}]')
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
    args = parser.parse_args()
    return args
