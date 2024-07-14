import os
import sys
sys.path.append(str(os.getcwd()))

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands

mlflow_exp_name = os.path.basename(__file__)

PROCESS_BATCH_SIZE = 1

# RUN_FILE = "parametricSN/main.py"
#RUN_FILE = "/data1/lxt/work4-2023/Ablation/parametricScatteringNetworks-vit/parametricSN/main_complex_single_vit_all.py"
RUN_FILE = "/mnt/data/lxt/work4-2023/Ablation/parametricScatteringNetworks-3090/parametricSN/main_complex_single_vit_all.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.1
DF = 25
THREE_PHASE = 1
LEARNABLE = 1
# EPOCHS = 1000
EPOCHS = 3500
INIT = "Tight-Frame"
RUNS_PER_SEED = 10
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 50000
# TRAIN_BATCH_SIZE = 1024
TRAIN_BATCH_SIZE = 1536
AUGMENT = "autoaugment"
# MODEL = "cnn"
MODEL = "vit"
PHASE_ENDS = " ".join(["100","200"])
MODEL_LOSS = 'cross-entropy'
SCATT_LRMAX = 0.2
SCATT_DF = 25
SCATT_THREE_PHASE = 1


if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []

    # for SEED in [207715039, 491659600,737523103,493572006,827192296,877498678,1103100946,1210393663,1277404878,1377264326]:
    for SEED in [207715039,491659600,737523103]:
        # for aa in [(1,"Random"),(1,"Tight-Frame"),(0,"Tight-Frame"),(0,"Random")]:
        for aa in [(1,"Tight-Frame")]:
            LEARNABLE, INIT = aa

            args1 = "-oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {}".format(
                OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM
            )

            args2 = "-os {} -daug {} -en {} -dtbs {} -mname {} -ope {}".format(
                SCHEDULER,AUGMENT,mlflow_exp_name,TRAIN_BATCH_SIZE,MODEL,PHASE_ENDS
            )

            args3 = "-smaxlr {} -sdivf {} -stp {} -mloss {}".format(
                SCATT_LRMAX,SCATT_DF,SCATT_THREE_PHASE,MODEL_LOSS
            )

            command = "{} {} run-train {} {} {} {}".format(
                PYTHON,RUN_FILE,args1,args2,args3,DATA_ARG)

            commands.append(command)

    
    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )






