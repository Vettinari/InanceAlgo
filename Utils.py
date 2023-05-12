import os
import sys
import time
import shutil
import pickle
import zipfile
from typing import Callable, Optional

import gc
import gzip
import os
from pathlib import Path

import datetime
import traceback
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool

from tqdm.auto import tqdm


def make_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def start_logging(filename=None) -> None:
    path = os.getcwd()
    today = datetime.today()
    new_path = make_path(f'{path}/logs/{str(today.date())}')
    if filename is None:
        filename = str(today.time())
    sys.stdout = open(f'{new_path}/{filename.split(".")[0]}.txt', "w")


def string_to_unix(date_string, date_format="%Y-%m-%d"):
    return time.mktime(datetime.strptime(date_string, date_format).timetuple())


def string_to_datetime(date_string, include_time=True):
    date_format = '%Y-%m-%d %H:%M:%S'
    if include_time:
        return datetime.strptime(date_string, date_format)
    else:
        return datetime.strptime(date_string, date_format).date()


def concurrent_execution(function, args):
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(function, args)


def concurrent_execution_with_return(function, args):
    with ThreadPoolExecutor() as executor:
        return executor.map(function, args)


def multiprocess_execution(function, args):
    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.map(function, args)


def get_config(api_type):
    api_type = api_type.upper()
    allowed_api_types = ['EOD', 'CRYPTO', 'FOREX', 'STOCK', 'COMMODITIES', 'TELEGRAM', 'TELEGRAM_ROOM', 'COMET']
    if api_type in allowed_api_types:
        try:
            config_file = open(f"/Users/milosz/Documents/Pycharm/Ainance/api_config", "r")
        except FileNotFoundError:
            raise Exception('Config file is missing.')

        file_array = config_file.read().split('\n')
        temp_dict = {}
        for line in file_array:
            line_array = line.split('=')
            temp_dict[line_array[0]] = line_array[1]
        return temp_dict[api_type]

    else:
        raise Exception('Chosen api_type is not supported.\n'
                        'Choose: "EOD", "CRYPTO", "FOREX", "STOCK" or "COMMODITIES"')


def save_object(object_to_save, path: str, filename: str):
    dir_path = make_path(path)
    full_path = f'{dir_path}/{filename}'
    with open(full_path, 'wb') as class_dump:
        pickle.dump(object_to_save, class_dump)


def load_object(path, filename: Optional[str] = None):
    # Create a variable to hold the contents of the object
    object_contents = b""
    if filename is None:
        obj = [obj for obj in os.listdir(path) if not obj.startswith(".")]
        filename = obj[0]

    # Set the chunk size to read from the file
    chunk_size = 1024 * 1024 * 500

    # Open the object using tqdm to display a progress bar
    with tqdm(total=os.path.getsize(os.path.join(path, filename)), unit="B", unit_scale=True, desc=filename) as pbar:
        # Open the file in binary mode
        with open(os.path.join(path, filename), "rb") as f:
            # Define a function to read a chunk of data from the file
            def read_chunk(start):
                f.seek(start)
                chunk = f.read(chunk_size)
                return chunk

            # Get the size of the file
            file_size = os.path.getsize(os.path.join(path, filename))

            # Create a thread pool with 8 threads
            with ThreadPoolExecutor(max_workers=8) as pool:
                # Create a list of start positions for each chunk
                starts = list(range(0, file_size, chunk_size))

                # Submit each chunk to the thread pool for reading
                futures = [pool.submit(read_chunk, start) for start in starts]

                # Iterate over the completed chunks and append them to the object contents
                for future in futures:
                    chunk = future.result()
                    object_contents += chunk
                    pbar.update(len(chunk))

    # Return the object contents
    pickle_object = pickle.loads(object_contents)
    return pickle_object


def find_exception_in_function(function, args):
    try:
        function(args)
    except Exception:
        traceback.print_exc()


def time_function(function: Callable, args):
    if args is None:
        start = time.time()
        function()
        end = round(time.time() - start, 5)
        print(f'{function.__name__} function took: {end} seconds | {round(end / 60, 2)} mins')
    else:
        start = time.time()
        function(args)
        end = round(time.time() - start, 5)
        print(f'{function.__name__} function took: {end} seconds | {round(end / 60, 2)} mins')


def printline(text, size=60, line_char="=", blank=False, title=False, test=True):
    if test is True:
        if blank is True:
            out = ''.join([line_char for i in range(size)])
            print(out[:size - 1])
        else:
            if title:
                text_size_with_spacing = len(text) + 2
                if text_size_with_spacing <= 70:
                    side_chars_amount = (size - text_size_with_spacing) // 2
                    side_print = ''.join([line_char for i in range(side_chars_amount)])
                    out = f"{side_print} {text} {side_print}"
                    out_line = ''.join(['-' for i in range(size)])
                    print(out_line[:size - 1])
                    print(out[:size - 1])
                    print(out_line[:size - 1])
            else:
                text_size_with_spacing = len(text) + 2
                if text_size_with_spacing <= 70:
                    side_chars_amount = (size - text_size_with_spacing) // 2
                    side_print = ''.join([line_char for i in range(side_chars_amount)])
                    out = f"{side_print} {text} {side_print}"
                    print(out[:size - 1])
                else:
                    print(text)
    else:
        pass


def window_to_array(window):
    return window.reshape(window.shape[0] * window.shape[1], )


def check_file(file_path):
    if os.path.isfile(file_path):
        return file_path
    else:
        return False


def move_dir_content(source_dir, destination_dir):
    if os.path.isdir(source_dir):
        destination_dir = make_path(destination_dir)

        if source_dir.endswith("/") is False:
            source_dir += "/"
        if destination_dir.endswith("/") is False:
            destination_dir += "/"

        # fetch all files
        for file_name in os.listdir(source_dir):
            # construct full file path
            source = source_dir + file_name
            destination = destination_dir + file_name
            # move only files
            if os.path.isfile(source):
                shutil.move(source, destination)
        os.rmdir(source_dir)
    else:
        print(f'Source directory:\n{source_dir} not found!')
