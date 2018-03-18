# WORD2VEC_FILE_PATH = "/home/h/data/word2vec/GoogleNews-vectors-negative300.bin.gz"
WORD2VEC_FILE_PATH  = "/home/h/data/word2vec/word2vec_text"


D_IN = 600
D_OUT = 300
D_HIDDEN = 300
HIDDEN_LAYERS = 1

BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
EPOCHS = 5
SAMPLE_FACTOR = 1 #number of samples to draw in each epoch
LR = 1e-4
LOG_INTERVAL = 100
TEST_INTERVAL = 10000

DATA_LOADER_WORKERS = 2

EVAL_TOP_K_ =5 #evaulate precision at rank for this value

THREADS = 2

CUDA_FLAG = True