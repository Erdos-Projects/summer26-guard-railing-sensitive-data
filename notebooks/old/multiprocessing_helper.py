from typing import Optional, Any
from multiprocessing import Pool
import polars as pl
import os


def linkage_attack(someone: pl.DataFrame, dataset: pl.DataFrame, quasi_ids: list[str]) -> pl.DataFrame:
    reidentified = someone.join(dataset, on=quasi_ids)
    return reidentified

def parallel_linkage_attack(args: list[Any], num_processes: Optional[int] = None) -> list[Any]:
    result = None
    n = os.cpu_count() if num_processes == None else num_processes
    with Pool(n) as p:
        result = p.starmap(linkage_attack, args)
    return result

def main():
    pass

if __name__ == "__main__":
    main()