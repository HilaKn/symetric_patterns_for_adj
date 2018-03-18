import argparse
from adj_predictor_runner import PredictorRunner
from data_handler import DataHandler

def run():
    data_handler = DataHandler(args.train_data_file, args.test_data_file)
    runner = PredictorRunner(data_handler, args.model_path, args.train_mode)
    runner.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare data for adjective prediction task')
    parser.add_argument('train_data_file',help='file containing adj-noun-adj triplets with sampling weights for training')
    parser.add_argument('model_path',help='path for the model pickle - in case of training this is an output path. in case of testing its from where to load the model')
    parser.add_argument('test_data_file',help='dataset for testing (extrinsic evaluation)')
    parser.add_argument('-t','--train_mode',  default=False, action='store_true',help='train mode or only test if False')

    args = parser.parse_args()
    run()