# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "python-dotenv",
#     "watchdog",
#     "rq",
#     "redis",
#     "mne",
#     "eeglabio",
# ]
# ///


# master.py
import os
import time
import multiprocessing
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rq import Worker, Queue, Connection
from redis import Redis
from tasks import sidecar_q, process_q, process_new_file, process_sidecar
import importlib
from tasks import clean_up_raw
from pathlib import Path

def start_rq_worker(queue_names, redis_host, redis_port, redis_db):
    """
    Starts an RQ worker listening to specified queues.
    """
    redis_conn = Redis(host=redis_host, port=redis_port, db=redis_db)
    queues = [Queue(name, connection=redis_conn) for name in queue_names]
    with Connection(redis_conn):
        worker = Worker(queues)
        worker.work()

class ConfigurableFileHandler(FileSystemEventHandler):
    """
    Watchdog event handler that enqueues tasks based on file extensions.
    """
    def __init__(self, extension, queue_name, task_function_name):
        super().__init__()
        self.extension = extension
        self.queue_name = queue_name
        self.task_function_name = task_function_name
        # Set up RQ queue
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_db = int(os.getenv('REDIS_DB', 0))
        self.redis_conn = Redis(host=redis_host, port=redis_port, db=redis_db)
        self.queue = Queue(self.queue_name, connection=self.redis_conn)
        # Import the task function
        # Assuming tasks.py is in the same directory
        tasks_module = importlib.import_module('tasks')
        self.task_function = getattr(tasks_module, self.task_function_name)
    
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(f'.{self.extension}'):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Detected new file: {event.src_path}")
            # Enqueue the corresponding task
            self.queue.enqueue(self.task_function, event.src_path)
            
    def on_deleted(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(f'.{self.extension}'):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Detected deleted file: {event.src_path}")
            # Enqueue the corresponding task
            self.queue.enqueue(clean_up_raw, event.src_path)

def start_watchdog_monitor(path, extension, queue_name, task_function_name):
    """
    Starts a Watchdog observer for a specific path and extension.
    """
    event_handler = ConfigurableFileHandler(extension, queue_name, task_function_name)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print(f"Started Watchdog monitor for .{extension} files in {path}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def main():
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    # Load environment variables
    load_dotenv()
    AUTOCLEAN_DIR = os.getenv('AUTOCLEAN_DIR')
    


    # Define watch configurations for new raw/set files
    new_file_watch_configs = [
        {"path": os.path.join(AUTOCLEAN_DIR, "resting_eyesopen", "watch"), 
         "extension": "raw", 
         "queue_name": "preprocessing", 
         "task_function_name": "autoclean_resting_eyesopen"},
    ]

    # Define watch configurations for sidecar JSON files
    postcomp_watch_configs = [
        {"path": os.path.join(AUTOCLEAN_DIR, "resting_eyesopen", "postcomp"), 
         "extension": "set", 
         "queue_name": "analysis", 
         "task_function_name": "analysis_resting_eyesopen"},
    ]

    # Combine all watch configurations
    all_watch_configs = new_file_watch_configs + postcomp_watch_configs

    # Ensure all watch directories exist
    for config in all_watch_configs:
        os.makedirs(config['path'], exist_ok=True)

    # Start RQ worker in a separate process
    worker_process = multiprocessing.Process(
        target=start_rq_worker, 
        args=(['preprocessing', 'analysis'], REDIS_HOST, REDIS_PORT, REDIS_DB),
        daemon=True
    )
    worker_process.start()
    print("RQ worker started and listening to 'preprocessing' and 'analysis' queues.")

    # Start Watchdog monitors in separate processes
    watchdog_processes = []
    for config in all_watch_configs:
        p = multiprocessing.Process(
            target=start_watchdog_monitor, 
            args=(
                config['path'], 
                config['extension'], 
                config['queue_name'], 
                config['task_function_name']
            ),
            daemon=True
        )
        p.start()
        watchdog_processes.append(p)
        print(f"Started Watchdog monitor for .{config['extension']} files in {config['path']}")

    print("All Watchdog monitors started.")
    print("System is running. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down system...")
        for p in watchdog_processes:
            p.terminate()
            p.join()
        worker_process.terminate()
        worker_process.join()
        print("System shutdown complete.")

if __name__ == "__main__":
    main()
