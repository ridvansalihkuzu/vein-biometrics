"""
This code is generated by Ridvan Salih KUZU @UNIROMA3
LAST EDITED:  01.04.2020
ABOUT SCRIPT:
It is a script for exporting the features of SDUMLA database images into CSV files as a list of features.
"""

import argparse
import torch
from models import DenseNet161_Modified as modelnet
from benchmark_verification import get_dataloader
from data_sdumla.utils import feature_exporter

parser = argparse.ArgumentParser(description='Vein Verification')

parser.add_argument('--num-classes', default=318, type=int, metavar='NC',
                    help='number of clases (default: 10000)')
parser.add_argument('--embedding-size', default=1024, type=int, metavar='ES',
                    help='embedding size (default: 128)')
parser.add_argument('--batch-size', default=256, type=int, metavar='BS',
                    help='batch size (default: 128)')
parser.add_argument('--num-workers', default=4, type=int, metavar='NW',
                    help='number of workers (default: 8)')
parser.add_argument('--main-model', default='modeldir/', type=str,
                    help='model directory (default: )')
parser.add_argument('--database-dir', default='data_sdumla/Database', type=str,
                    help='path to the database root directory')
parser.add_argument('--train-val-indir', default='data_sdumla/CSVFiles/output_list_train_val.csv', type=str,
                    help='path to the CSV input file including the list of train/validation partitions')
parser.add_argument('--test-indir', default='data_sdumla/CSVFiles/output_list_test.csv', type=str,
                    help='path to the CSV input file including the list of test partition')
parser.add_argument('--train-val-feat_outdir', default='data_sdumla/CSVFiles/encoded_feature_list_test_sparse-01.csv', type=str,
                    help='path to the CSV output file including the list of train/validation features')
parser.add_argument('--test-feat_outdir', default='data_sdumla/CSVFiles/encoded_feature_list_train_val_sparse-01.csv', type=str,
                    help='path to the CSV output file including the list of test features')
args = parser.parse_args()

def main():

    export_all(args.train_val_indir, args.test_indir, args.train_val_feat_outdir, args.test_feat_outdir)


def export_all(train_valid_file_list, test_file_list, train_valid_features, test_features):
    """It extracts the features of the whole images given in a dataset:
       Args:
           TRAIN_VALID_FILE_LIST: The CSV file to read train files for extracting their featues
           TEST_FILE_LIST: The CSV file to read test files for extracting their features
           TRAIN_VALID_FEATURES: The output CSV file to write the extracted features from train_file_list
           TEST_FEATURES: The output CSV file to write the extracted features from valid_file_list
    """
    data_loaders = get_dataloader(args.database_dir, train_valid_file_list, test_file_list, test_file_list,
                                  args.batch_size, args.num_workers)
    train_loader = data_loaders['train']
    test_loader = data_loaders['valid']

    main_model = modelnet(embedding_size=args.embedding_size, pretrained=True)
    main_model = torch.nn.DataParallel(main_model).cuda()
    checkpoint = torch.load(args.main_model + '/model_best.pth.tar')
    main_model.load_state_dict(checkpoint['state_dict'])


    feature_exporter(main_model, train_loader, train_valid_features)
    feature_exporter(main_model, test_loader, test_features)


if __name__ == '__main__':
    main()