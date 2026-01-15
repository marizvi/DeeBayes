import os

# Signal Parameters
SAMPLE_SIZE = 1024
SNR_LEVEL = 5  # Desired SNR for noise addition

# Training Hyperparameters
BATCH_SIZE = 128
N_EPOCHS = 75
LEARNING_RATE = 0.0002

# VDN Specific Parameters
RADIUS = 3
WIN = 2 * RADIUS + 1
SIGMA_SPATIAL = RADIUS
EPS2 = 1e-6

# Paths
DATA_PATH = './data'  # Path to MIT-BIH and NSTDB files
CHECKPOINT_PATH = './models/VDN-main-1024-bsize128-es16-ma-em-5.pt'

# Records from MIT-BIH
TRAIN_RECORDS = ['100','101','102','104','106','107','108','109','112','113','114',
                 '115','117','118','119','121','123','124','200','201','202','203',
                 '207','208','209','210','212','214','215','217','220','221','222',
                 '228','231','232','233','234']

TEST_RECORDS = ['103','105','111','116','122','205','213','219','223','230']