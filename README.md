# Local MySQL: Build, Start, and Stop

This guide covers `/ssd_root/he923/mysql-server`, based on its source code, existing CMake caches, and `/home/he923/run_stop_mysql/*.sh`. The source version is **MySQL 9.4.0**, with InnoDB vector indexes and FAISS, HNSWLIB, and DiskANN backends. The original upstream documentation is preserved in [README](README).

## 1. Choose the correct instance

| Purpose | Build directory | Data directory | Original startup script |
| --- | --- | --- | --- |
| Debug, for development | `/ssd_root/he923/mysql-server/build` | `/ssd_root/he923/mysqldata` | `~/run_stop_mysql/start_my_mysql.sh` |
| Release, for performance testing | `/ssd_root/he923/mysql-server/build-release` | `/ssd_root/he923/mysqldata_rls` | `~/run_stop_mysql/rls_start_my_mysql.sh` |
| Separate MyVector project | `/ssd_root/he923/mysql_myvector/mysql-server/build` | `/ssd_root/he923/mysql_myvector/mysqldata` | `~/run_stop_mysql/myvector_start.sh` |

The first two scripts both use **TCP port `33077`, socket `/tmp/mysql.sock`, and an 8G InnoDB buffer pool**. These instances cannot run simultaneously with their original configurations. MyVector uses port `33078` but shares `/tmp/mysql.sock`, so it still conflicts at the socket level. It belongs to a separate source tree.

Both data directories for this project already contain databases. **Do not reinitialize or clear them.** If the executables already exist, proceed directly to startup. Build only when updating source code or when a rebuild is needed.

## 2. Start an existing database

Run the following commands as the login user `he923`; `sudo` is not required. Choose one pair of variables in the same Bash terminal used for subsequent commands.

```bash
# Debug
MYSQL_BUILD=/ssd_root/he923/mysql-server/build
MYSQL_DATA=/ssd_root/he923/mysqldata
```

Or:

```bash
# Release
MYSQL_BUILD=/ssd_root/he923/mysql-server/build-release
MYSQL_DATA=/ssd_root/he923/mysqldata_rls
```

Before starting, check existing processes and port/socket usage:

```bash
pgrep -a mysqld
ss -ltnp | rg ':33077\b'
ss -lxnp | rg '/tmp/mysql.sock'
```

If another instance uses the selected data directory, port, or socket, identify it and shut it down normally first. Do not delete socket or PID files that are still in use.

Start the server:

```bash
"$MYSQL_BUILD/runtime_output_directory/mysqld" \
  --no-defaults \
  --basedir="$MYSQL_BUILD" \
  --datadir="$MYSQL_DATA" \
  --socket=/tmp/mysql.sock \
  --port=33077 \
  --user=he923 \
  --bind-address=127.0.0.1 \
  --mysqlx=OFF \
  --log-error="$MYSQL_DATA/mysqld.err" \
  --innodb-buffer-pool-size=8G \
  --max-allowed-packet=1G \
  --pid-file="$MYSQL_DATA/mysqld.pid" \
  --daemonize
```

This keeps the original data paths and SQL port, runs in the background with `--daemonize`, restricts TCP access to the local machine, and disables MySQL X Protocol. Adjust the corresponding options if remote access or X Protocol is needed. On machines with less memory, reduce `8G` to `1G` or `2G`.

Place `--no-defaults` first to avoid reading the system `my.cnf`. It does not prevent loading persisted variables from `mysqld-auto.cnf` in the data directory. `--basedir` points to the build directory; `--datadir` points to the database files.

Check the startup log and connect:

```bash
tail -n 80 "$MYSQL_DATA/mysqld.err"

"$MYSQL_BUILD/runtime_output_directory/mysql" \
  --no-defaults --protocol=SOCKET --socket=/tmp/mysql.sock -u root -p
```

In MySQL, confirm which instance you reached:

```sql
SELECT VERSION(), @@version_comment, @@basedir, @@datadir, @@port, @@socket;
SELECT 1;
```

Check `@@datadir` in particular. Several scripts share `/tmp/mysql.sock`, so a successful connection does not guarantee that you reached the intended database.

To explicitly use TCP:

```bash
"$MYSQL_BUILD/runtime_output_directory/mysql" \
  --no-defaults --protocol=TCP -h 127.0.0.1 -P 33077 -u root -p
```

`-p` prompts for the database password. Restarting an existing database does not change its password. For a new instance created with `--initialize-insecure` whose password has not yet been set, omit `-p`. The database password is separate from the Linux login password.

### Using the original scripts

After confirming that no instance conflicts and that stopping the system MySQL service is acceptable, choose one:

```bash
bash ~/run_stop_mysql/start_my_mysql.sh       # Debug
# Or:
bash ~/run_stop_mysql/rls_start_my_mysql.sh   # Release
```

Both scripts run `sudo systemctl stop mysql`, directly remove `/tmp/mysql.sock` and their respective PID files, then launch the server with `&`. They do not wait for the database to become ready. The message `Starting custom MySQL...` therefore does not prove a successful startup; check the log and run SQL as shown above. Prefer the manual commands for routine operation.

## 3. Shut down and restart normally

First verify the instance using the SQL query above, then run:

```bash
"$MYSQL_BUILD/runtime_output_directory/mysqladmin" \
  --no-defaults --protocol=SOCKET --socket=/tmp/mysql.sock -u root -p shutdown

tail -n 40 "$MYSQL_DATA/mysqld.err"
```

Wait for `Shutdown complete` in the log and for the corresponding process to exit before running the startup command again.

If shutdown through a database account is unavailable, read the PID from the selected data directory, verify the process, and send `SIGTERM`:

```bash
MYSQL_PID=$(cat "$MYSQL_DATA/mysqld.pid")
ps -p "$MYSQL_PID" -o pid,user,args
# Verify that the process above belongs to MYSQL_DATA before running:
kill -TERM "$MYSQL_PID"
```

The original shutdown scripts are `stop_my_mysql.sh` and `rls_stop_my_mysql.sh`. They wait only 5 seconds after sending a signal, then use `kill -9` if the process is still running. This can interrupt flushing and require crash recovery on the next startup, so they are not the preferred method for routine shutdown. A normal shutdown usually requires no manual removal of PID or socket files.

## 4. Build environment and dependencies

The following environment was verified on this machine:

| Component | Local version / requirement |
| --- | --- |
| Operating system | Ubuntu 22.04, x86_64 |
| Compiler | GCC/G++ 11.4.0; this repository's Linux configuration requires GCC 11 or later |
| CMake | `/usr/local/bin/cmake`, version 3.27.0; the bundled FAISS requires at least 3.24 |
| Generator | Unix Makefiles |
| OpenSSL | System OpenSSL 3.0.2 |
| Vector backends | `WITH_VECINDEX`, `WITH_FAISS`, `WITH_HNSWLIB`, and `WITH_DISKANN` are all ON |

Check the toolchain:

```bash
/usr/local/bin/cmake --version
/usr/bin/g++ --version
bison --version
```

The dependencies are already installed on this machine. On another machine running the same Ubuntu version, use the following package list as a starting point and check the actual CMake results:

```bash
sudo apt-get update
sudo apt-get install build-essential bison pkg-config \
  libssl-dev libncurses-dev libaio-dev libnuma-dev \
  libtirpc-dev rpcsvc-proto \
  libboost-dev libboost-program-options-dev \
  libmkl-dev libomp-dev libgoogle-perftools-dev
```

CMake is intentionally absent from this apt command: ensure that the available version is at least 3.24. This guide uses the locally installed 3.27.0. Compression libraries, Protobuf, ICU, and other dependencies use bundled source versions.

FAISS uses MKL/BLAS. DiskANN also requires Boost program_options, MKL, and OpenMP. The current DiskANN configuration searches for MKL/OMP libraries under `/usr/lib/x86_64-linux-gnu` and headers under `/usr/include/mkl`. For nonstandard installations, specify `-DMKL_PATH=... -DMKL_INCLUDE_PATH=... -DOMP_PATH=...`.

Check that external source directories are present:

```bash
cd /ssd_root/he923/mysql-server
git submodule status
ls external/faiss/CMakeLists.txt external/hnswlib/CMakeLists.txt \
   external/DiskANN/mysql/CMakeLists.txt
```

FAISS and HNSWLIB are Git submodules. Run `git submodule update --init --recursive` only for a fresh clone whose submodules have not been initialized. The current FAISS submodule has local modifications; do not overwrite or reset them. DiskANN is not listed in the root `.gitmodules`, so ensure that `external/DiskANN` is included when moving the source tree.

## 5. Incremental builds using existing build directories

After changing source code, shut down the instance whose binaries will be updated, then choose one build:

```bash
# Debug
/usr/local/bin/cmake --build /ssd_root/he923/mysql-server/build \
  --target mysqld mysql mysqladmin --parallel 8
```

```bash
# Release
/usr/local/bin/cmake --build /ssd_root/he923/mysql-server/build-release \
  --target mysqld mysql mysqladmin --parallel 8
```

These commands build the server, connection/administration clients, and their dependencies. They avoid building Router, all tests, and all third-party tools. The parallelism of `8` is an example; reduce it to `2` or `4` if memory is limited. To build all default targets, omit `--target mysqld mysql mysqladmin`.

Neither `make install` nor `sudo make install` is required. The startup scripts use the build outputs directly:

```text
build[-release]/runtime_output_directory/mysqld
build[-release]/runtime_output_directory/mysql
build[-release]/runtime_output_directory/mysqladmin
build[-release]/library_output_directory/    # Runtime shared libraries
build[-release]/share/                      # Error messages and other resources
```

Do not keep only the `mysqld` executable and delete the rest of the build directory: it depends on shared libraries and resources in that tree. Restart the instance after building so that the new process uses the updated code. Database initialization is not required again.

## 6. Initial configuration or rebuilding a build directory

The existing caches use `build: Debug + WITH_DEBUG=1` and `build-release: Release + WITH_DEBUG=OFF`. When configuration is needed, run the corresponding commands from the source directory.

### Debug

```bash
cd /ssd_root/he923/mysql-server
/usr/local/bin/cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DWITH_DEBUG=ON \
  -DWITH_SSL=system \
  -DWITH_CURL=none \
  -DWITH_VECINDEX=ON \
  -DWITH_FAISS=ON \
  -DWITH_HNSWLIB=ON \
  -DWITH_DISKANN=ON

/usr/local/bin/cmake --build build \
  --target mysqld mysql mysqladmin --parallel 8
```

### Release

```bash
cd /ssd_root/he923/mysql-server
/usr/local/bin/cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_DEBUG=OFF \
  -DWITH_SSL=system \
  -DWITH_CURL=none \
  -DWITH_VECINDEX=ON \
  -DWITH_FAISS=ON \
  -DWITH_HNSWLIB=ON \
  -DWITH_DISKANN=ON

/usr/local/bin/cmake --build build-release \
  --target mysqld mysql mysqladmin --parallel 8
```

The root CMake configuration disables FAISS GPU support, Python bindings, and the C API. This CPU build does not require CUDA. `WITH_CURL=none` matches the existing local caches.

**CPU instruction sets:** `storage/innobase/vec/CMakeLists.txt` currently prefers the `faiss_avx512` target whenever it exists. FAISS defines that target even with `FAISS_OPT_LEVEL=generic`, so changing that option alone does not guarantee a server without AVX-512 instructions. This machine's Xeon Gold 6330 supports AVX-512. Check and adjust target selection before moving to another CPU. DiskANN also enables AVX2 and defaults to `-march=native` in Release builds.

## 7. First initialization: new, empty data directories only

Skip this section for the existing `mysqldata` and `mysqldata_rls` directories. The example uses a separate new directory to keep existing data intact.

```bash
MYSQL_BUILD=/ssd_root/he923/mysql-server/build-release
MYSQL_DATA=/ssd_root/he923/mysqldata_new

# Do not use mkdir -p: skip initialization if the directory already exists.
if mkdir -m 700 "$MYSQL_DATA"; then
  "$MYSQL_BUILD/runtime_output_directory/mysqld" \
    --no-defaults \
    --initialize-insecure \
    --basedir="$MYSQL_BUILD" \
    --datadir="$MYSQL_DATA" \
    --user=he923
else
  printf 'Directory already exists or could not be created; initialization skipped. Check the path.\n' >&2
fi
```

After successful initialization, keep this `MYSQL_DATA` value and follow section 2 to start the server. Omit `-p` for the first connection. `--initialize-insecure` creates `root@localhost` with an empty initial password. After connecting locally, you can set a password interactively:

```sql
ALTER USER 'root'@'localhost' IDENTIFIED BY 'replace-with-your-own-password';
```

Use `-p` for subsequent connections and shutdowns. The original startup scripts point to the old data directories and will not automatically use `mysqldata_new`.

## 8. Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Can't connect ... /tmp/mysql.sock` | Read `mysqld.err` in the selected data directory. Check whether the process exited and whether the socket path matches. |
| Connected to the wrong database | Query `@@datadir`. Several scripts share a socket; use TCP with an explicit target port if needed. |
| `Address already in use` / socket in use | Identify the existing instance. Concurrent instances need separate data directories, sockets, ports, PID files, and logs; disable X Protocol or give it separate settings too. |
| `Unable to lock ... ibdata1` | Check for another process using the same data directory. Do not delete InnoDB files. |
| `Access denied` | Use the database credentials for that data directory. Do not reinitialize it to fix a login issue. |
| CMake version too old | Use `/usr/local/bin/cmake` and verify that the version is at least 3.24. |
| Missing MKL / Intel OMP / Boost | Check the dependencies and library paths in section 4. |
| `cc1plus` killed | Check for insufficient memory and reduce `--parallel`. |
| `Illegal instruction` | Check whether the CPU supports the instructions used by FAISS/DiskANN; see section 6. |
| Missing shared library or `errmsg.sys` | Keep the complete build directory and check `--basedir`. Use `ldd .../mysqld` to look for `not found`. |
| InnoDB assertion / signal 6 in the log | Analyze the latest crash log. Historical Debug logs contain such entries, but those alone do not establish the current state of the data. |

## 9. Verification scope

During preparation of this guide, the startup/shutdown scripts, source configuration, and both build caches were inspected. The server executables reported `9.4.0-debug` and `9.4.0`, respectively, and no missing runtime libraries were found for Release. Both servers passed `--validate-config` checks for the main startup options shown here.

The Release configuration options in section 6 were verified in the separate temporary directory `/tmp/mysql-readme-cmake.eAymDG`. CMake configuration and generation succeeded; the log is `configure.log` in that directory. All 17 Bash code blocks passed `bash -n` syntax checks. A complete source build was not performed.

Preparing this guide did not start, stop, or reinitialize existing databases or rebuild their binaries. Configuration validation does not prove that the server can start or recover an existing database successfully; confirm those steps using the startup log and SQL queries.
