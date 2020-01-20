# Author: K.Degiorgio
#
# project wide configurations

import os
import logging
import shutil
import logging
import torch
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_id = datetime.now().strftime("%d.%H.%M")
out_dir = os.path.join("out", run_id)
__logger_setup = False


def create_dir(directory, clean=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif clean:
        shutil.rmtree(directory)
        os.makedirs(directory)


def get_save_file(name):
    return os.path.join(out_dir, name)


def getlogger(area):
    global __logger_setup
    if not __logger_setup:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M",
            filename=os.path.join(out_dir, "app.log"),
            filemode="w",
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)
        __logger_setup = True
    return logging.getLogger(area)


def logging_debug():
    logging.basicConfig(level=logging.DEBUG)


create_dir(out_dir)
