import os
import sys
import json
import logging

import yaml
import numpy as np

DEV_MODE = True  # simplify the flpackage re-setup everytime we change the source codes of flpackage
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_logger
from federatedscope.config import cfg, assert_cfg
from federatedscope.core.DAIL_fed_api import DAILFed
#from flpackage.core.DAIL_fed_api_single_process import DAILFed_SIM
from federatedscope.autotune import split_raw_config, config2cmdargs, get_scheduler

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    setup_logger(cfg)

    args = parse_args()
    with open(args.cfg_file, 'r') as ips:
        config = yaml.load(ips, Loader=yaml.FullLoader)
    det_config, tbd_config = split_raw_config(config)
    cfg.merge_from_list(config2cmdargs(det_config))
    cfg.merge_from_list(args.opts)
    assert_cfg(cfg)

    scheduler = get_scheduler(tbd_config)
    results = scheduler.optimize()
    logging.info(results)
