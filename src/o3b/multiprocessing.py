import logging

logger = logging.getLogger(__name__)
import os
import torch
from tqdm import tqdm
from torch.utils.data import get_worker_info
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

def get_num_workers_suggested(factor=0.5):
    # max(torch.cuda.device_count() * 4, 1)
    # max(os.cpu_count() - 1, 1) # number of cores available for system
    # max(len(os.sched_getaffinity(0)) - 1, 1) # number of cores available for current process    
    return max(int(len(os.sched_getaffinity(0)) * factor), 1)

def get_ram_in_gb():
    import psutil
    ram_in_bytes = psutil.virtual_memory()
    ram_in_gb = int(round(ram_in_bytes.total / (1024**3)))
    return ram_in_gb

    #print("Total RAM:", memory.total / (1024**3), "GB")
    #print("Available RAM:", memory.available / (1024**3), "GB")
    #print("Used RAM:", memory.used / (1024**3), "GB")
    #print("RAM usage percentage:", memory.percent, "%")


class SQLTableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        db_url,
        table_name,
        chunk_size=10_000,
        batch_size=10,
        row_fn=None,
        read_total_row_count=False
    ):
        super().__init__()
        self.db_url = db_url
        self.table_name = table_name
        self.chunk_size = chunk_size
        self.row_fn = row_fn
        if read_total_row_count:
            logger.info("creating sql engine...")
            # imported here, not at module scope: sqlalchemy is only needed by
            # SQLTableDataset, and a top-level import made o3b.cv.io — which
            # every video-backed loader goes through — unimportable without it.
            from sqlalchemy import create_engine
            engine = create_engine(self.db_url)
            logger.info("read total rows...")
            from sqlalchemy import text
            with engine.connect() as conn:
                total_rows = conn.execute(
                    text(f"SELECT COUNT(*) FROM {self.table_name}")
                ).scalar()

                logger.info(f"total number of rows are {total_rows}, and {round(total_rows / batch_size, 1)} batches")

        # query = f"SELECT COUNT(*) FROM {self.table_name}"
        # logger.info("read dataframe from sql count table...")
        # df = pd.read_sql(query, engine)
        # logger.info("read total rows from dataframe...")
        # total_rows = df.iloc[0, 0]

    def __iter__(self):
        from sqlalchemy import create_engine

        worker_info = get_worker_info()
        engine = create_engine(self.db_url)

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        for i, chunk in enumerate(
            pd.read_sql_table(
                self.table_name,
                engine,
                chunksize=self.chunk_size,
            )
        ):
            if i % num_workers != worker_id:
                continue

            #for row in chunk.itertuples(index=False): 
            #    # itertuples faster than iterrows but requires row.attribute instead of row["attribute"]
            #    # but requires that columns are valid identifiers. E.g. starting with underscore (_) does not work.
            #    row_as_series = pd.Series(row._asdict())
            
            for _, row in chunk.iterrows():
                if self.row_fn is None:
                    row
                else:
                    self.row_fn(row)

                yield [0]


def apply_sql_table_per_row(sql_db_url, table_name, row_fn, chunk_size=1_000, batch_size=10, num_workers=None, prefetch_factor=None, 
                            pin_memory=False, persistent_workers=True):

    if num_workers is None:
        num_workers = get_num_workers_suggested()
            
    dataset = SQLTableDataset(
        db_url=sql_db_url, # "postgresql://user:password@localhost/mydb",
        table_name=table_name, # "frame_annots",
        chunk_size=chunk_size,
        batch_size=batch_size,
        row_fn=row_fn,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    for _ in tqdm(loader):
        pass

def get_subdirs_at_depth(root, target_depth):
    """
    Return all directories at exactly `target_depth` below root.
    Depth 0 => root itself
    """
    result = []

    def walk(current_path, depth):
        if depth == target_depth:
            result.append(current_path)
            return

        try:
            with os.scandir(current_path) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        walk(entry.path, depth + 1)
        except FileNotFoundError:
            pass

    walk(root, 0)
    return result

def delete_tree(path):
    shutil.rmtree(path, ignore_errors=True)
    return path

def rm_in_parallel(path, max_depth=0):
    for target_depth_inv in range(max_depth+1):
        target_depth = max_depth - target_depth_inv
        subdirs = get_subdirs_at_depth(root=path, target_depth=target_depth)

        max_workers = get_num_workers_suggested()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(delete_tree, d) for d in subdirs]

            for f in as_completed(futures):
                logger.info(f"Deleted: {f.result()}")


def parallel_items_with_item_fn(items, item_fn, spawn=False):
    import multiprocessing as mp
    ctx = mp.get_context("spawn" if spawn else "fork") # forkserver

    with ProcessPoolExecutor(max_workers=get_num_workers_suggested(), mp_context=ctx) as executor:
        results = list(
            tqdm(
                executor.map(item_fn, items),
                total=len(items)
            )
        )