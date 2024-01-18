import pandas as pd
import argparse
import asyncio
import json
import time
import sys
import os
import glob
from api import *
from utils import (
    get_config_attrs, JSONEncoder, create_dirs, cmd_request_type,
    write_collected_chats
)
from processjson import process_json

# Parsing setup
parser = argparse.ArgumentParser(description='Arguments.')
parser.add_argument(
    '--telegram-channel',
    type=str,
    required=False,
    help='Username of a telegram channel.'
)
parser.add_argument(
    '--batch-file',
    type=str,
    required=False,
    help='Path to a file with a list of telegram channels, should have one per line.'
)
parser.add_argument(
    '--limit-download-to-channel-metadata',
    action='store_true',
    help='Then only the channel metadata will be downloaded, not the messages.'
)
parser.add_argument(
    '--output',
    '-o',
    type=str,
    required=False,
    help='Folder to save collected data. Default: `./output/data`'
)
parser.add_argument(
    '--min-id',
    type=int,
    help='Minimum message ID to start downloading from. Default: 0'
)
parser.add_argument(
    '--process-messages',
    action='store_true',
    help='Process *_messages.json files in the output directory after main operations.'
)

args = vars(parser.parse_args())
config_attrs = get_config_attrs()
args = {**args, **config_attrs}

if not args['process_messages']:
    if not args['telegram_channel'] and not args['batch_file']:
        parser.error('Either --telegram-channel or --batch-file must be provided unless --process-messages is specified.')

# log results
text = f'''
Init program at {time.ctime()}
'''
print(text)

# Function to process messages files
def process_messages_files(output_dir):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_folder = os.path.join(repo_root, 'dataset')

    for subdir in glob.glob(f"{output_dir}/*/"):
        for file in glob.glob(f"{subdir}*_messages.json"):
            try:
                channel_name = os.path.basename(subdir[:-1])
                processed_file = os.path.join(dataset_folder, f"{channel_name}_processed.json")
                process_json(file, processed_file)
                print(f"Processed file saved: {processed_file}")
            except Exception as e:
                print(f"Error processing file {file}: {e}")

if args['process_messages']:
    output_folder = args['output'] if args['output'] else './output/data'
    process_messages_files(output_folder)
else:
    # Existing logic for telegram channel processing
    sfile = 'session_file'
    api_id = args['api_id']
    api_hash = args['api_hash']
    phone = args['phone']

    loop = asyncio.get_event_loop()
    client = loop.run_until_complete(
        get_connection(sfile, api_id, api_hash, phone)
    )

    req_type, req_input = cmd_request_type(args)
    if req_type == 'batch':
        req_input = [i.rstrip() for i in open(req_input, encoding='utf-8', mode='r')]
    else:
        req_input = [req_input]

    if args['output']:
        output_folder = args['output'].rstrip('/')
    else:
        output_folder = './output/data'

    create_dirs(output_folder)

    # ... [Rest of your main script logic for telegram channel processing] ...

# log results
text = f'''
End program at {time.ctime()}
'''
print(text)
