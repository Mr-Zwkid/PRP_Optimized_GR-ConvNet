import logging

import torch

logging.basicConfig(level=logging.INFO)


def get_device(force_cpu):
    # Check if CUDA can be used
    if torch.cuda.is_available() and not force_cpu:
        logging.info("CUDA detected. Running with GPU acceleration.")
        device = "cuda"
    elif force_cpu:
        logging.info("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
        device = "cpu"  # torch.device("cpu")
    else:
        logging.info("CUDA is *NOT* detected. Running with only CPU.")
        device = "cpu"
    return device


def delay_time():
    # compensate hardware performance
    return 0


def jacquard(results):
    # compensate Jacquard
    results['correct'] = results['correct']
    results['failed'] = results['failed']
    return results


def cornell(results):
    # compensate Cornell
    return results


def test():
    the_ds_shuffle = False
    return the_ds_shuffle
