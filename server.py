import os
import time
import multiprocessing
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rq import Worker, Queue, Connection
from redis import Redis
from tasks import autoclean_rest_eyesopen, analysis_rest_eyesopen, clean_up_raw, autoclean_chirp_default
from pathlib import Path

# Load environment variables
load_dotenv()

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
    def __init__(self, extension_to_task):
        super().__init__()
        # Mapping from file extensions to (queue_name, task_function)
        self.extension_to_task = extension_to_task
        # Set up RQ connections
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_db = int(os.getenv('REDIS_DB', 0))
        self.redis_conn = Redis(host=redis_host, port=redis_port, db=redis_db)
        # Prepare queues
        self.queues = {}
        for queue_name in set(q for q, _ in extension_to_task.values()):
            self.queues[queue_name] = Queue(queue_name, connection=self.redis_conn)
    
    def on_created(self, event):
        if event.is_directory:
            return
        _, ext = os.path.splitext(event.src_path)
        ext = ext.lstrip('.')
        if ext in self.extension_to_task:
            queue_name, task_function = self.extension_to_task[ext]
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Detected new file: {event.src_path}")
            self.queues[queue_name].enqueue(task_function, event.src_path)
    
    def on_deleted(self, event):
        if event.is_directory:
            return
        _, ext = os.path.splitext(event.src_path)
        ext = ext.lstrip('.')
        if ext in self.extension_to_task:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Detected deleted file: {event.src_path}")
            self.queues['preprocessing'].enqueue(clean_up_raw, event.src_path)

def main():
    

    # Load environment variables
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    AUTOCLEAN_DIR = os.getenv('AUTOCLEAN_DIR')
    
    # Define watch configurations
    watch_configs = [
        {
            "path": os.path.join(AUTOCLEAN_DIR, "rest_eyesopen", "watch"),
            "extension_to_task": {
                "raw": ("preprocessing", autoclean_rest_eyesopen),
            }
        },
        {
            "path": os.path.join(AUTOCLEAN_DIR, "chirp_default", "watch"),
            "extension_to_task": {
                "raw": ("preprocessing", autoclean_chirp_default),
            }
        },
        {
            "path": os.path.join(AUTOCLEAN_DIR, "rest_eyesopen", "postcomps"),
            "extension_to_task": {
                "ready": ("analysis", analysis_rest_eyesopen),
            }
        },
    ]
    
    # Ensure all watch directories exist
    for config in watch_configs:
        os.makedirs(config['path'], exist_ok=True)
    
        # Set the start method for multiprocessing
    multiprocessing.set_start_method("spawn")  # Alternatives: "spawn" or "forkserver"

    # Start RQ worker in a separate process
    worker_process = multiprocessing.Process(
        target=start_rq_worker, 
        args=(['preprocessing', 'analysis'], REDIS_HOST, REDIS_PORT, REDIS_DB),
        daemon=True
    )
    worker_process.start()
    print("RQ worker started and listening to 'preprocessing' and 'analysis' queues.")
    
    # Start Watchdog observer
    observer = Observer()
    for config in watch_configs:
        event_handler = ConfigurableFileHandler(config['extension_to_task'])
        observer.schedule(event_handler, config['path'], recursive=False)
        print(f"Started Watchdog monitor for path {config['path']}")

    observer.start()
    print("All Watchdog monitors started.")
    print("System is running. Press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down system...")
        observer.stop()
        observer.join()
        worker_process.terminate()
        worker_process.join()
        print("System shutdown complete.")

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

def display_environment(console):
    env_vars = {
        "AUTOCLEAN_DIR": os.getenv('AUTOCLEAN_DIR'),
        "UNPROCESSED_DIR": os.getenv('UNPROCESSED_DIR'),
        "REDIS_HOST": os.getenv('REDIS_HOST'),
        "REDIS_PORT": os.getenv('REDIS_PORT'),
        "REDIS_DB": os.getenv('REDIS_DB'),
    }

    table = Table(title="Environment Variables", show_header=True, header_style="bold magenta")
    table.add_column("Variable", style="dim", width=20)
    table.add_column("Value", style="bold cyan")

    for key, value in env_vars.items():
        table.add_row(key, value)

    console.print(table)

if __name__ == "__main__":
    console = Console()

    console.print(Panel.fit("[bold green]Starting Autoclean REST Eyes Open Service[/bold green]", 
                            title="Service Status", border_style="green"))

    display_environment(console)

    with console.status("[bold green]Running main process...", spinner="dots") as status:
        try:
            main()
            console.print("[bold green]Main process completed successfully![/bold green]")
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")
            raise
