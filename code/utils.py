import gzip

class Utils:

    def load_to_list(self,file_name):
        '''
        read file lines to list
        :param file_name:
        :return: list of rows
        '''
        with open (file_name) as f:
            lines = [line.rstrip('\n') for line in f]

        return lines




