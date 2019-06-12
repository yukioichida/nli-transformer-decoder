#!/usr/bin/env python3

import logging


def get_logger(identifier):
    logger = logging.getLogger(identifier)
    hdlr = logging.FileHandler('logs/{}.log'.format(identifier))
    console_log = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    hdlr.setFormatter(formatter)
    console_log.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.addHandler(console_log)
    logger.setLevel(logging.INFO)
    return logger
