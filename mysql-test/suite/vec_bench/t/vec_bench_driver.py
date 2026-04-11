#!/usr/bin/env python3

import argparse
import csv
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import mysql.connector
import numpy as np


DEFAULT_EF_SEARCH = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176]
DEFAULT_QUERY_COUNT = 10_000
DEFAULT_TOPK = 100
DEFAULT_INSERT_BATCH = 10_000
DEFAULT_BUILD_THREADS = 128
DEFAULT_SCHEMA = "vec_bench"
DEFAULT_HNSW_M = 32
DEFAULT_EF_CONSTRUCTION = 128
BARRIER_TIMEOUT_S = 300


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    table_name: str
    index_name: str
    base_kind: str
    base_path: Path
    query_path: Path
    gt_path: Path
    dim: int
    rows: int
    metric: str = "l2"


@dataclass
class BenchRow:
    mode: str
    dataset: str
    topk: int
    ef_search: int
    query_count: int
    connections: int
    recall: float
    avg_latency_ms: float
    qps: float
    total_time_s: float


class Worker(threading.Thread):
    def __init__(
        self,
        runner: "VecBenchRunner",
        dataset: DatasetSpec,
        query_payloads: Sequence[bytes],
        gt_sets: Sequence[Set[int]],
        query_indices: Sequence[int],
        topk: int,
        options: str,
        ready_barrier: threading.Barrier,
        start_event: threading.Event,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._dataset = dataset
        self._query_payloads = query_payloads
        self._gt_sets = gt_sets
        self._query_indices = query_indices
        self._topk = topk
        self._options = options
        self._ready_barrier = ready_barrier
        self._start_event = start_event
        self.correct = 0
        self.latency_sum = 0.0
        self.error: Optional[BaseException] = None

    def run(self) -> None:
        conn = None
        cursor = None
        try:
            conn = self._runner.connect(database=self._runner.args.schema, autocommit=True)
            cursor = conn.cursor()
            sql = (
                f"SELECT id FROM `{self._runner.args.schema}`.`{self._dataset.table_name}` "
                f"WHERE MYVECTOR_IS_ANN(v, %s, %s) LIMIT {self._topk}"
            )

            self._ready_barrier.wait(timeout=BARRIER_TIMEOUT_S)
            self._start_event.wait()

            for idx in self._query_indices:
                start = time.perf_counter()
                cursor.execute(sql, (self._query_payloads[idx], self._options))
                rows = cursor.fetchall()
                self.latency_sum += time.perf_counter() - start
                found = {int(row[0]) for row in rows[: self._topk]}
                self.correct += len(found & self._gt_sets[idx])
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()


class VecBenchRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.report_dir = Path(args.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = self._build_datasets()

    def _build_datasets(self) -> List[DatasetSpec]:
        return [
            DatasetSpec(
                name="sift10m",
                table_name="sift10m",
                index_name="idx_sift10m_v",
                base_kind="bvecs",
                base_path=Path(self.args.sift_base),
                query_path=Path(self.args.sift_query),
                gt_path=Path(self.args.sift_gt),
                dim=128,
                rows=10_000_000,
            ),
            DatasetSpec(
                name="deep10m",
                table_name="deep10m",
                index_name="idx_deep10m_v",
                base_kind="fbin",
                base_path=Path(self.args.deep_base),
                query_path=Path(self.args.deep_query),
                gt_path=Path(self.args.deep_gt),
                dim=96,
                rows=10_000_000,
            ),
        ]

    def connect(self, database: Optional[str], autocommit: bool) -> mysql.connector.MySQLConnection:
        params: Dict[str, object] = {
            "user": self.args.user,
            "autocommit": autocommit,
        }
        if database:
            params["database"] = database
        if self.args.socket:
            params["unix_socket"] = self.args.socket
        else:
            params["host"] = self.args.host
            params["port"] = self.args.port
        return mysql.connector.connect(**params)

    def ensure_schema(self) -> None:
        conn = self.connect(database=None, autocommit=True)
        cursor = conn.cursor()
        try:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.args.schema}`")
            cursor.execute(f"USE `{self.args.schema}`")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_state (
                    dataset_name VARCHAR(64) PRIMARY KEY,
                    table_name VARCHAR(64) NOT NULL,
                    row_count BIGINT NOT NULL,
                    dim INT NOT NULL,
                    build_threads INT NOT NULL,
                    status VARCHAR(16) NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
                """
            )
        finally:
            cursor.close()
            conn.close()

    def mark_state(self, dataset: DatasetSpec, status: str) -> None:
        conn = self.connect(database=self.args.schema, autocommit=True)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO benchmark_state (
                    dataset_name, table_name, row_count, dim, build_threads, status
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    table_name = VALUES(table_name),
                    row_count = VALUES(row_count),
                    dim = VALUES(dim),
                    build_threads = VALUES(build_threads),
                    status = VALUES(status)
                """,
                (
                    dataset.name,
                    dataset.table_name,
                    dataset.rows,
                    dataset.dim,
                    self.args.build_threads,
                    status,
                ),
            )
        finally:
            cursor.close()
            conn.close()

    def dataset_is_ready(self, dataset: DatasetSpec) -> bool:
        conn = self.connect(database=self.args.schema, autocommit=True)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT status, row_count, dim, build_threads
                FROM benchmark_state
                WHERE dataset_name = %s
                """,
                (dataset.name,),
            )
            row = cursor.fetchone()
            if row is None:
                return False
            status, row_count, dim, build_threads = row
            if status != "READY":
                return False
            if int(row_count) != dataset.rows or int(dim) != dataset.dim:
                return False
            if int(build_threads) != self.args.build_threads:
                return False
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
                """,
                (self.args.schema, dataset.table_name),
            )
            return int(cursor.fetchone()[0]) == 1
        finally:
            cursor.close()
            conn.close()

    def prepare_datasets(self) -> None:
        self.ensure_schema()
        for dataset in self.datasets:
            self._validate_dataset_files(dataset)
            if self.dataset_is_ready(dataset):
                print(f"[prepare] Reusing {dataset.name}")
                continue
            self.prepare_dataset(dataset)

    def _validate_dataset_files(self, dataset: DatasetSpec) -> None:
        for path in (dataset.base_path, dataset.query_path, dataset.gt_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required dataset file does not exist: {path}")

    def prepare_dataset(self, dataset: DatasetSpec) -> None:
        print(f"[prepare] Loading {dataset.name} from {dataset.base_path}")
        self.mark_state(dataset, "LOADING")
        conn = self.connect(database=self.args.schema, autocommit=False)
        cursor = conn.cursor()
        try:
            cursor.execute(f"DROP TABLE IF EXISTS `{dataset.table_name}`")
            cursor.execute(
                f"""
                CREATE TABLE `{dataset.table_name}` (
                    id BIGINT NOT NULL,
                    v VECTOR({dataset.dim}) NOT NULL,
                    PRIMARY KEY (id)
                ) ENGINE=InnoDB
                """
            )
            conn.commit()

            insert_sql = (
                f"INSERT INTO `{dataset.table_name}` (id, v) VALUES (%s, UNHEX(%s))"
            )
            total_loaded = 0
            next_id = 1

            for matrix in self.iter_base_batches(dataset, self.args.insert_batch_size):
                rows = self.vector_batch_to_rows(matrix, start_id=next_id)
                cursor.executemany(insert_sql, rows)
                conn.commit()
                loaded_now = len(rows)
                total_loaded += loaded_now
                next_id += loaded_now
                if total_loaded % max(self.args.insert_batch_size * 10, 1) == 0 or total_loaded == dataset.rows:
                    print(f"[prepare] {dataset.name}: loaded {total_loaded}/{dataset.rows}")

            if total_loaded != dataset.rows:
                raise RuntimeError(
                    f"Loaded {total_loaded} rows for {dataset.name}, expected {dataset.rows}"
                )

            comment = (
                f"type=hnsw,backend=hnswlib,metric={dataset.metric},dim={dataset.dim},"
                f"size={dataset.rows},threads={self.args.build_threads},"
                f"hnsw_m={DEFAULT_HNSW_M},ef={DEFAULT_EF_CONSTRUCTION}"
            )
            print(f"[prepare] Building vector index for {dataset.name} with {self.args.build_threads} threads")
            cursor.execute(
                f"""
                ALTER TABLE `{dataset.table_name}`
                ADD VECINDEX `{dataset.index_name}` (v)
                COMMENT '{comment}'
                """
            )
            conn.commit()
        except BaseException:  # noqa: BLE001
            conn.rollback()
            self.mark_state(dataset, "FAILED")
            raise
        finally:
            cursor.close()
            conn.close()

        self.verify_vecindex(dataset)
        self.mark_state(dataset, "READY")

    def iter_base_batches(self, dataset: DatasetSpec, batch_size: int) -> Iterator[np.ndarray]:
        if dataset.base_kind == "bvecs":
            yield from iter_bvecs_batches(dataset.base_path, dataset.dim, batch_size, dataset.rows)
            return

        matrix = open_fbin_matrix(dataset.base_path, dataset.dim)
        if matrix.shape[0] < dataset.rows:
            raise RuntimeError(
                f"{dataset.base_path} only contains {matrix.shape[0]} vectors, expected {dataset.rows}"
            )
        for start in range(0, dataset.rows, batch_size):
            end = min(start + batch_size, dataset.rows)
            yield np.asarray(matrix[start:end], dtype=np.float32)

    def vector_batch_to_rows(self, matrix: np.ndarray, start_id: int) -> List[Tuple[int, str]]:
        float_matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        byte_matrix = float_matrix.view(np.uint8).reshape(float_matrix.shape[0], float_matrix.shape[1] * 4)
        return [
            (start_id + offset, byte_matrix[offset].tobytes().hex())
            for offset in range(float_matrix.shape[0])
        ]

    def verify_vecindex(self, dataset: DatasetSpec) -> None:
        queries = self.load_query_payloads(dataset, self.args.query_count)
        if not queries:
            raise RuntimeError(f"No queries loaded for {dataset.name}")
        conn = self.connect(database=self.args.schema, autocommit=True)
        cursor = conn.cursor(dictionary=True)
        try:
            sql = (
                f"EXPLAIN SELECT id FROM `{self.args.schema}`.`{dataset.table_name}` "
                f"WHERE MYVECTOR_IS_ANN(v, %s, %s) LIMIT {self.args.topk}"
            )
            options = f"query_type=binary,k={self.args.topk},efSearch={self.args.ef_search[0]}"
            cursor.execute(sql, (queries[0], options))
            rows = cursor.fetchall()
        finally:
            cursor.close()
            conn.close()

        explain_types = {str(row.get("type", "")).lower() for row in rows}
        if "vecindex" not in explain_types:
            raise RuntimeError(
                f"Vector index is not used for {dataset.name}; EXPLAIN types={sorted(explain_types)}"
            )

    def load_query_payloads(self, dataset: DatasetSpec, query_count: int) -> List[bytes]:
        matrix = load_query_matrix(dataset)
        if matrix.shape[0] < query_count:
            raise RuntimeError(
                f"{dataset.query_path} only contains {matrix.shape[0]} queries, expected {query_count}"
            )
        float_matrix = np.ascontiguousarray(matrix[:query_count], dtype=np.float32)
        byte_matrix = float_matrix.view(np.uint8).reshape(float_matrix.shape[0], float_matrix.shape[1] * 4)
        return [byte_matrix[idx].tobytes() for idx in range(float_matrix.shape[0])]

    def load_groundtruth_sets(
        self,
        dataset: DatasetSpec,
        topk: int,
        query_count: int,
    ) -> List[Set[int]]:
        gt = load_ivecs(dataset.gt_path)
        if gt.shape[0] < query_count:
            raise RuntimeError(
                f"{dataset.gt_path} only contains {gt.shape[0]} groundtruth rows, expected {query_count}"
            )
        if gt.shape[1] < topk:
            raise RuntimeError(
                f"{dataset.gt_path} only provides top-{gt.shape[1]} neighbors, requested top-{topk}"
            )
        return [
            set((gt[idx, :topk] + 1).tolist())
            for idx in range(query_count)
        ]

    def run(self) -> List[BenchRow]:
        self.prepare_datasets()
        rows: List[BenchRow] = []
        for dataset in self.datasets:
            query_payloads = self.load_query_payloads(dataset, self.args.query_count)
            gt_sets = self.load_groundtruth_sets(dataset, self.args.topk, self.args.query_count)
            if self.args.mode == "latency":
                rows.extend(self.run_latency(dataset, query_payloads, gt_sets))
            else:
                rows.extend(self.run_throughput(dataset, query_payloads, gt_sets))
        return rows

    def run_latency(
        self,
        dataset: DatasetSpec,
        query_payloads: Sequence[bytes],
        gt_sets: Sequence[Set[int]],
    ) -> List[BenchRow]:
        print(f"[latency] Running {dataset.name}")
        conn = self.connect(database=self.args.schema, autocommit=True)
        cursor = conn.cursor()
        sql = (
            f"SELECT id FROM `{self.args.schema}`.`{dataset.table_name}` "
            f"WHERE MYVECTOR_IS_ANN(v, %s, %s) LIMIT {self.args.topk}"
        )
        rows: List[BenchRow] = []

        try:
            for ef_search in self.args.ef_search:
                options = f"query_type=binary,k={self.args.topk},efSearch={ef_search}"
                total_hits = 0
                latency_sum = 0.0
                wall_start = time.perf_counter()
                for idx, query_payload in enumerate(query_payloads):
                    start = time.perf_counter()
                    cursor.execute(sql, (query_payload, options))
                    result_rows = cursor.fetchall()
                    latency_sum += time.perf_counter() - start
                    found = {int(row[0]) for row in result_rows[: self.args.topk]}
                    total_hits += len(found & gt_sets[idx])
                total_time = time.perf_counter() - wall_start
                query_total = len(query_payloads)
                rows.append(
                    BenchRow(
                        mode="latency",
                        dataset=dataset.name,
                        topk=self.args.topk,
                        ef_search=ef_search,
                        query_count=query_total,
                        connections=self.args.latency_connections,
                        recall=total_hits / (query_total * self.args.topk),
                        avg_latency_ms=(latency_sum / query_total) * 1000.0,
                        qps=query_total / total_time,
                        total_time_s=total_time,
                    )
                )
                print(
                    f"[latency] {dataset.name} efSearch={ef_search} "
                    f"recall={rows[-1].recall:.6f} avg_latency_ms={rows[-1].avg_latency_ms:.4f}"
                )
        finally:
            cursor.close()
            conn.close()

        return rows

    def run_throughput(
        self,
        dataset: DatasetSpec,
        query_payloads: Sequence[bytes],
        gt_sets: Sequence[Set[int]],
    ) -> List[BenchRow]:
        print(f"[throughput] Running {dataset.name}")
        rows: List[BenchRow] = []

        query_indices = split_indices(len(query_payloads), self.args.throughput_connections)
        for ef_search in self.args.ef_search:
            options = f"query_type=binary,k={self.args.topk},efSearch={ef_search}"
            ready_barrier = threading.Barrier(self.args.throughput_connections + 1)
            start_event = threading.Event()
            workers = [
                Worker(
                    runner=self,
                    dataset=dataset,
                    query_payloads=query_payloads,
                    gt_sets=gt_sets,
                    query_indices=query_indices[idx],
                    topk=self.args.topk,
                    options=options,
                    ready_barrier=ready_barrier,
                    start_event=start_event,
                )
                for idx in range(self.args.throughput_connections)
            ]

            for worker in workers:
                worker.start()

            barrier_error: Optional[BaseException] = None
            wall_start = 0.0
            try:
                ready_barrier.wait(timeout=BARRIER_TIMEOUT_S)
            except threading.BrokenBarrierError as exc:
                barrier_error = exc
            else:
                wall_start = time.perf_counter()
                start_event.set()

            for worker in workers:
                worker.join()
                if worker.error is not None:
                    raise worker.error
            if barrier_error is not None:
                raise RuntimeError("Timed out while waiting for throughput workers to initialize") from barrier_error

            total_time = time.perf_counter() - wall_start

            total_hits = sum(worker.correct for worker in workers)
            latency_sum = sum(worker.latency_sum for worker in workers)
            query_total = len(query_payloads)
            rows.append(
                BenchRow(
                    mode="throughput",
                    dataset=dataset.name,
                    topk=self.args.topk,
                    ef_search=ef_search,
                    query_count=query_total,
                    connections=self.args.throughput_connections,
                    recall=total_hits / (query_total * self.args.topk),
                    avg_latency_ms=(latency_sum / query_total) * 1000.0,
                    qps=query_total / total_time,
                    total_time_s=total_time,
                )
            )
            print(
                f"[throughput] {dataset.name} efSearch={ef_search} "
                f"recall={rows[-1].recall:.6f} qps={rows[-1].qps:.2f}"
            )

        return rows

    def write_reports(self, rows: Sequence[BenchRow]) -> None:
        csv_path = self.report_dir / f"{self.args.report_prefix}_report.csv"
        json_path = self.report_dir / f"{self.args.report_prefix}_report.json"
        manifest_path = self.report_dir / f"{self.args.report_prefix}_manifest.txt"

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "mode",
                    "dataset",
                    "topk",
                    "ef_search",
                    "query_count",
                    "connections",
                    "recall",
                    "avg_latency_ms",
                    "qps",
                    "total_time_s",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row.mode,
                        row.dataset,
                        row.topk,
                        row.ef_search,
                        row.query_count,
                        row.connections,
                        f"{row.recall:.8f}",
                        f"{row.avg_latency_ms:.8f}",
                        f"{row.qps:.8f}",
                        f"{row.total_time_s:.8f}",
                    ]
                )

        payload = {
            "mode": self.args.mode,
            "schema": self.args.schema,
            "topk": self.args.topk,
            "query_count": self.args.query_count,
            "connections": (
                self.args.latency_connections
                if self.args.mode == "latency"
                else self.args.throughput_connections
            ),
            "build_threads": self.args.build_threads,
            "ef_search": self.args.ef_search,
            "datasets": [dataset.name for dataset in self.datasets],
            "rows": [
                {
                    "mode": row.mode,
                    "dataset": row.dataset,
                    "topk": row.topk,
                    "ef_search": row.ef_search,
                    "query_count": row.query_count,
                    "connections": row.connections,
                    "recall": row.recall,
                    "avg_latency_ms": row.avg_latency_ms,
                    "qps": row.qps,
                    "total_time_s": row.total_time_s,
                }
                for row in rows
            ],
        }
        with json_path.open("w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, indent=2, sort_keys=True)
            json_file.write("\n")

        lines = [
            f"mode={self.args.mode}",
            f"schema={self.args.schema}",
            f"topk={self.args.topk}",
            f"query_count={self.args.query_count}",
            (
                f"connections={self.args.latency_connections}"
                if self.args.mode == "latency"
                else f"connections={self.args.throughput_connections}"
            ),
            f"build_threads={self.args.build_threads}",
            "ef_search=" + ",".join(str(value) for value in self.args.ef_search),
            "datasets=" + ",".join(dataset.name for dataset in self.datasets),
            f"csv={csv_path}",
            f"json={json_path}",
            f"csv_rows={len(rows)}",
            f"cleanup_after={1 if self.args.cleanup_after else 0}",
        ]
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            manifest_file.write("\n".join(lines))
            manifest_file.write("\n")

    def cleanup(self) -> None:
        conn = self.connect(database=None, autocommit=True)
        cursor = conn.cursor()
        try:
            cursor.execute(f"DROP DATABASE IF EXISTS `{self.args.schema}`")
        finally:
            cursor.close()
            conn.close()


def iter_bvecs_batches(path: Path, dim: int, batch_size: int, expected_rows: int) -> Iterator[np.ndarray]:
    record_dtype = np.dtype([("dim", "<i4"), ("vec", np.uint8, (dim,))])
    loaded = 0
    with path.open("rb") as handle:
        while loaded < expected_rows:
            take = min(batch_size, expected_rows - loaded)
            chunk = np.fromfile(handle, dtype=record_dtype, count=take)
            if chunk.size == 0:
                break
            dims = chunk["dim"]
            if not np.all(dims == dim):
                unique_dims = sorted(set(int(value) for value in dims.tolist()))
                raise RuntimeError(f"{path} contains unexpected dimensions: {unique_dims}")
            loaded += int(chunk.shape[0])
            yield chunk["vec"].astype(np.float32)
    if loaded != expected_rows:
        raise RuntimeError(f"{path} contains {loaded} vectors, expected {expected_rows}")


def load_query_matrix(dataset: DatasetSpec) -> np.ndarray:
    if dataset.base_kind == "bvecs":
        record_dtype = np.dtype([("dim", "<i4"), ("vec", np.uint8, (dataset.dim,))])
        data = np.fromfile(dataset.query_path, dtype=record_dtype)
        if data.size == 0:
            raise RuntimeError(f"{dataset.query_path} does not contain any query vectors")
        dims = data["dim"]
        if not np.all(dims == dataset.dim):
            unique_dims = sorted(set(int(value) for value in dims.tolist()))
            raise RuntimeError(f"{dataset.query_path} contains unexpected dimensions: {unique_dims}")
        return data["vec"].astype(np.float32)
    return np.asarray(open_fbin_matrix(dataset.query_path, dataset.dim), dtype=np.float32)


def open_fbin_matrix(path: Path, expected_dim: int) -> np.memmap:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=2)
    if header.size != 2:
        raise RuntimeError(f"{path} is missing the fbin header")
    row_count, dim = int(header[0]), int(header[1])
    if dim != expected_dim:
        raise RuntimeError(f"{path} has dim={dim}, expected {expected_dim}")
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(row_count, dim))


def load_ivecs(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        raise RuntimeError(f"{path} does not contain any groundtruth rows")
    width = int(data[0])
    if width <= 0:
        raise RuntimeError(f"{path} has an invalid row width: {width}")
    if data.size % (width + 1) != 0:
        raise RuntimeError(f"{path} size is not aligned to ivecs rows")
    rows = data.reshape(-1, width + 1)
    if not np.all(rows[:, 0] == width):
        raise RuntimeError(f"{path} contains rows with inconsistent widths")
    return rows[:, 1:]


def split_indices(total: int, parts: int) -> List[List[int]]:
    base, remainder = divmod(total, parts)
    result: List[List[int]] = []
    start = 0
    for part in range(parts):
        stop = start + base + (1 if part < remainder else 0)
        result.append(list(range(start, stop)))
        start = stop
    return result


def parse_positive_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        value = int(text)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"Expected positive integers, got {raw}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("At least one efSearch value is required")
    return values


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vector benchmark runner for mysql-test")
    parser.add_argument("--mode", choices=("latency", "throughput"), required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--socket", default="")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--report-prefix", required=True)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--ef-search", type=parse_positive_int_list, default=DEFAULT_EF_SEARCH)
    parser.add_argument("--insert-batch-size", type=int, default=DEFAULT_INSERT_BATCH)
    parser.add_argument("--build-threads", type=int, default=DEFAULT_BUILD_THREADS)
    parser.add_argument("--latency-connections", type=int, default=1)
    parser.add_argument("--throughput-connections", type=int, default=32)
    parser.add_argument("--cleanup-after", type=int, choices=(0, 1), default=0)
    parser.add_argument("--sift-base", required=True)
    parser.add_argument("--sift-query", required=True)
    parser.add_argument("--sift-gt", required=True)
    parser.add_argument("--deep-base", required=True)
    parser.add_argument("--deep-query", required=True)
    parser.add_argument("--deep-gt", required=True)
    args = parser.parse_args(argv)

    if args.topk <= 0:
        parser.error("--topk must be positive")
    if args.query_count <= 0:
        parser.error("--query-count must be positive")
    if args.insert_batch_size <= 0:
        parser.error("--insert-batch-size must be positive")
    if args.build_threads <= 0:
        parser.error("--build-threads must be positive")
    if args.latency_connections <= 0:
        parser.error("--latency-connections must be positive")
    if args.throughput_connections <= 0:
        parser.error("--throughput-connections must be positive")

    return args


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    runner = VecBenchRunner(args)
    success = False
    try:
        rows = runner.run()
        runner.write_reports(rows)
        success = True
    finally:
        if success and args.cleanup_after:
            runner.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
