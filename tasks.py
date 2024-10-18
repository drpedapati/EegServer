# tasks.py
import os
from redis import Redis
from rq import Queue
from dotenv import load_dotenv
from pathlib import Path
import time
from autoclean_rest_eyesopen import entrypoint as do_autoclean_rest_eyesopen

# Load environment variables
load_dotenv()
BASE_DIR = os.getenv('AUTOCLEAN_DIR')

# Initialize Redis connection
redis_conn = Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=int(os.getenv('REDIS_PORT', 6379)), db=int(os.getenv('REDIS_DB', 0)))

# Initialize RQ queues
preprocessing_q = Queue('preprocessing', connection=redis_conn)
analysis_q = Queue('analysis', connection=redis_conn)

def autoclean_rest_eyesopen(file_path):
    """
    Task to process a new file by creating its sidecar JSON.
    """
    print("=" * 50)
    print("STARTING AUTOCLEAN REST EYES OPEN TASK")
    print("=" * 50)
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing new file: {file_path}")
        unprocessed_file = file_path
        eeg_system = "EGI128_RAW"
        task = "rest_eyesopen"
        config_file = Path("lossless_config_rest_eyesopen.yaml")
        do_autoclean_rest_eyesopen(unprocessed_file, eeg_system, task, config_file)
    except Exception as e:
        print(f"Error processing new file {file_path}: {e}")
        raise
    print("=" * 50)
    print("FINISHED AUTOCLEAN REST EYES OPEN TASK")
    print("=" * 50)

def analysis_rest_eyesopen(file_path):
    """
    Task to process a new file by creating its sidecar JSON.
    """
    print("=" * 50)
    print("STARTING ANALYSIS REST EYES OPEN TASK")
    print("=" * 50)
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing new file: {file_path}")
    except Exception as e:
        print(f"Error processing new file {file_path}: {e}")
        raise
    print("=" * 50)
    print("FINISHED ANALYSIS REST EYES OPEN TASK")
    print("=" * 50)
    
def autoclean_chirp_default(file_path):
    """
    Task to process a new file by creating its sidecar JSON.
    """
    print("=" * 50)
    print("STARTING AUTOCLEAN CHIRP DEFAULT TASK")
    print("=" * 50)
    
def clean_up_raw(file_path):
    """
    Task to process a new file by creating its sidecar JSON.
    """
    print("=" * 50)
    print("STARTING CLEAN UP RAW TASK")
    print("=" * 50)
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing new file: {file_path}")
    except Exception as e:
        print(f"Error processing new file {file_path}: {e}")
        raise
    print("=" * 50)
    print("FINISHED CLEAN UP RAW TASK")
    print("=" * 50)