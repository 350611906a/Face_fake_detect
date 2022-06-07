#coding=utf-8

import os
import json
import logging


def list_to_file(alist, fpath, postfix=''):
    """Save a list to a file with one element as a line."""
    assert isinstance(alist, list)
    assert isinstance(fpath, str)
    assert isinstance(postfix, str)

    if len(alist) == 0:
        print('Empty list.')
        return
    
    if fpath is None:
        print('Invalid file path.')
        return

    with open(fpath, 'w') as f:
        for a in alist:
            f.write(a + postfix + '\n')


def file_to_list(fpath):
    """Make a list from a file with one line as an item."""
    assert osp.exists(fpath)
    alist = []
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            alist += line.splitlines()
        return alist


def dict_to_file(adict, fpath):
    """Save a dict to file."""
    assert isinstance(adict, dict)
    assert isinstance(fpath, str)

    with open(fpath, 'w') as f:
        if fpath.endswith('.json'):
            json.dump(adict, f, indent=4, separators=(',', ':'))
        # elif fpath.endswith('.txt'):
        #     _, values = dict_to_list(adict)
        #     list_to_file(values, fpath)
        else:
            pickle.dump(adict, f)
    print('Dictionary is saved in {}.'.format(fpath))



def set_logger(filename, level=logging.INFO, logger_name=None):
    """Set a file logger while keep the console logging output."""
    logger = logging.getLogger(logger_name) 
    logger.setLevel(level)
    
    # Never mutate (insert/remove elements) the list you're currently iterating on. 
    # If you need, make a copy.
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
        # FileHandler is subclass of StreamHandler, so isinstance(handler,
        # logging.StreamHandler) is True even if handler is FileHandler.
        # if (type(handler) == logging.StreamHandler) and (handler.stream == sys.stderr):
        elif type(handler) == logging.StreamHandler:
            logger.removeHandler(handler)
            
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    return logger


def log_info(msg, info=None, is_debug=True):
    # TODO: add check code, only list and 1D ndarray are supported.
    # Not that common!
    # assert isinstance(info, list)

    # TODO: find which to use according to logging info.
    func = logging.debug if is_debug else logging.info
    if info is None:
        func(msg)
    else:
        output = [str(a) for a in info]
        output = ','.join(output)
        logging.debug('{}: {}'.format(msg, output))
        func('{}: {}'.format(msg, output))
