# Shared setup for vector benchmark tests.

--source include/big_test.inc
--source include/have_nodebug.inc
--source include/not_asan.inc
--source include/not_valgrind.inc
--source include/not_windows.inc

--let $PYTHON= python3
if (`SELECT CONVERT(@@VERSION_COMPILE_OS USING latin1) IN ('Win32', 'Win64', 'Windows')`)
{
  --let $PYTHON= python
}

--let $VEC_BENCH_SCHEMA= vec_bench
--let $VEC_BENCH_TOPK= 100
--let $VEC_BENCH_QUERY_COUNT= 10000
--let $VEC_BENCH_EF_LIST= 16,32,48,64,80,96,112,128,144,160,176
--let $VEC_BENCH_BUILD_THREADS= 128
--let $VEC_BENCH_LATENCY_CONNECTIONS= 1
--let $VEC_BENCH_THROUGHPUT_CONNECTIONS= 32

--let $VEC_BENCH_SIFT_BASE= /ssd_root/he923/mysql_dataset/sift10m/bigann_10m_base.bvecs
--let $VEC_BENCH_SIFT_QUERY= /ssd_root/he923/mysql_dataset/sift10m/bigann_query.bvecs
--let $VEC_BENCH_SIFT_GT= /ssd_root/he923/mysql_dataset/sift10m/idx_10M.ivecs
--let $VEC_BENCH_DEEP_BASE= /ssd_root/he923/mysql_dataset/deep10m/base.10M.fbin
--let $VEC_BENCH_DEEP_QUERY= /ssd_root/he923/mysql_dataset/deep10m/query.public.10K.fbin
--let $VEC_BENCH_DEEP_GT= /ssd_root/he923/mysql_dataset/deep10m/deep10M_groundtruth.ivecs

--file_exists $VEC_BENCH_SIFT_BASE
--file_exists $VEC_BENCH_SIFT_QUERY
--file_exists $VEC_BENCH_SIFT_GT
--file_exists $VEC_BENCH_DEEP_BASE
--file_exists $VEC_BENCH_DEEP_QUERY
--file_exists $VEC_BENCH_DEEP_GT

--let $MYSQL_SOCKET= `SELECT @@socket`
