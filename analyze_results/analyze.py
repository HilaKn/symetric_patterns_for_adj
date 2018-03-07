import argparse
from triplets_handler import   TripletHandler


def run():
    handler = TripletHandler(args.input_folder, args.output_file)
    handler.run()
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze patterns output - extract adj-adj-noun triplets.')
    parser.add_argument('input_folder',help='folder containing results from running patterns_code')
    parser.add_argument('output_file',help='output_file_name')
    args = parser.parse_args()
    run()