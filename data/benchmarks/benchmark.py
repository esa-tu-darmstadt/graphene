import subprocess
import threading
import queue
import time
import os
import math


def run_program(config, matrix, output_file, profiling_dir, numTiles, program_path):
    # Create the profiling directory if given
    profiling_flag = ""
    if profiling_dir:
        os.makedirs(profiling_dir, exist_ok=True)
        profiling_flag = "-d" + profiling_dir

    try:
        with open(output_file, "w") as out_file:
            print(f"Running {config} {matrix} with {numTiles} tiles")
            process = subprocess.run(
                [
                    program_path,
                    config,
                    matrix,
                    "-t" + str(numTiles),
                    profiling_flag,
                ],
                stdout=out_file,
                stderr=subprocess.PIPE,
                text=True,
            )
            print(f"Finished {config} {matrix}")
            if process.returncode != 0:
                print(f"Error for {config} {matrix}: {process.stderr}")
                return False
            return True
    except Exception as e:
        print(f"Exception for {config} {matrix}: {e}")
        return False


def worker(program_path, task_queue, ipu_semaphore):
    while not task_queue.empty():
        config, matrix, output_file, profiling_dir, numTiles = task_queue.get()
        ipus_needed = math.ceil(numTiles / 1472)
        # Round up to the nearest multiple of 4
        ipus_needed = 4 * math.ceil(ipus_needed / 4)
        with ipu_semaphore(ipus_needed):
            successfull = run_program(
                config, matrix, output_file, profiling_dir, numTiles, program_path
            )
            # Reschedule the task if it failed
            if not successfull:
                task_queue.put((config, matrix, output_file, profiling_dir, numTiles))
                print(f"Rescheduled {config} {matrix} with {numTiles} tiles")
        task_queue.task_done()


class IPUSemaphore:
    def __init__(self, total_ipus):
        self.total_ipus = total_ipus
        self.available_ipus = total_ipus
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def __call__(self, ipus_needed):
        return self.IPUContext(self, ipus_needed)

    class IPUContext:
        def __init__(self, semaphore, ipus_needed):
            self.semaphore = semaphore
            self.ipus_needed = ipus_needed

        def __enter__(self):
            with self.semaphore.condition:
                while self.semaphore.available_ipus < self.ipus_needed:
                    self.semaphore.condition.wait()
                self.semaphore.available_ipus -= self.ipus_needed

        def __exit__(self, exc_type, exc_val, exc_tb):
            with self.semaphore.condition:
                self.semaphore.available_ipus += self.ipus_needed
                self.semaphore.condition.notify_all()


def main():
    log_dir = "logs/convergence_plot/"
    config_files = [
        # "ir_gs.jsonc",
        # "spmv.jsonc",
        # "ilu0.jsonc",
        # "ir_pbicgstab_ilu0.jsonc",
        "ir_pbicgstab_ilu0_verbose_nomixedprecision.jsonc",
        # "ir_pbicgstab_ilu0_verbose.jsonc",
        # "restarter_pbicgstab_ilu0_verbose.jsonc",
    ]
    config_files_that_require_profiling = ["spmv.jsonc", "ilu0.jsonc"]
    matrices = [
        "-m../matrices/Hook_1498/Hook_1498.mtx",
        "-m../matrices/G3_circuit/G3_circuit.mtx",
        "-m../matrices/Geo_1438/Geo_1438.mtx",
        "-m../matrices/af_shell7/af_shell7.mtx",
        # "-p150,150,150",
        # "-p200,200,200",
    ]

    # for numRows in range(1000000, 12000000, 1000000):
    #     nx = (numRows) ** (1 / 3)
    #     nx = int(nx)
    #     conf = f"-p{nx},{nx},{nx}"
    #     matrices.append(conf)
    # rowsPerTile = 200 * 200 * 200 / 1472

    # tiles = range(1 * 1472, 17 * 1472, 1472)
    # tiles = [1472 * 10, 1472 * 13, 1472 * 14]
    # tiles = [20608]
    tiles = [5888]

    program_path = "../../build/applications/benchmark/benchmark"

    task_queue = queue.Queue()
    for numTiles in tiles:
        for config in config_files:
            ## For the weak scaling
            # nx = int((rowsPerTile * numTiles) ** (1 / 3))
            # matrices = [f"-p{nx},{nx},{nx}"]
            for matrix in matrices:
                config_name = config.split(".")[0]
                if matrix.startswith("-m"):
                    matrix_name = matrix.lstrip("-m").split("/")[-1].split(".")[0]
                elif matrix.startswith("-p"):
                    poissonConfig = matrix.lstrip("-p").split(",")
                    matrix_name = f"poisson_{poissonConfig[0]}x{poissonConfig[1]}x{poissonConfig[2]}"
                profiling_dir = ""
                if config in config_files_that_require_profiling:
                    profiling_dir = f"{log_dir}/profiling/{config_name}_{matrix_name}_{numTiles}tiles"
                output_file = (
                    f"{log_dir}/{config_name}_{matrix_name}_{numTiles}tiles.txt"
                )
                task_queue.put((config, matrix, output_file, profiling_dir, numTiles))

    total_ipus = 16
    ipu_semaphore = IPUSemaphore(total_ipus)
    max_workers = (
        total_ipus  # Set the number of worker threads to the total number of IPUs
    )

    threads = []
    for _ in range(max_workers):
        thread = threading.Thread(
            target=worker, args=(program_path, task_queue, ipu_semaphore)
        )
        thread.start()
        threads.append(thread)

    task_queue.join()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
