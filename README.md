# 本机 MySQL：编译、启动与停止

本文针对 `/ssd_root/he923/mysql-server`，根据源码、现有 CMake 缓存和 `/home/he923/run_stop_mysql/*.sh` 整理。源码版本为 **MySQL 9.4.0**，包含 InnoDB 向量索引及 FAISS、HNSWLIB、DiskANN 后端。原上游说明保留在 [README](README)。

## 1. 先选对实例

| 用途 | 构建目录 | 数据目录 | 原启动脚本 |
| --- | --- | --- | --- |
| Debug，开发调试 | `/ssd_root/he923/mysql-server/build` | `/ssd_root/he923/mysqldata` | `~/run_stop_mysql/start_my_mysql.sh` |
| Release，性能测试 | `/ssd_root/he923/mysql-server/build-release` | `/ssd_root/he923/mysqldata_rls` | `~/run_stop_mysql/rls_start_my_mysql.sh` |
| 另一套 MyVector 项目 | `/ssd_root/he923/mysql_myvector/mysql-server/build` | `/ssd_root/he923/mysql_myvector/mysqldata` | `~/run_stop_mysql/myvector_start.sh` |

前两套原脚本都使用 **TCP 端口 `33077`、socket `/tmp/mysql.sock`、8G InnoDB buffer pool**。它们不能按原配置同时启动。MyVector 的端口是 `33078`，但也使用 `/tmp/mysql.sock`，仍会发生 socket 冲突；它不属于本文的源码目录。

前两套数据目录目前都已有数据库文件，**不要重新初始化或清空**。已有可执行文件时可以直接跳到启动步骤；仅在修改源码或需要重新构建时编译。

## 2. 启动已有数据库

以下命令以登录用户 `he923` 执行，不需要 `sudo`。先在同一个 Bash 终端选择一组变量，后续命令复用它们。

```bash
# Debug
MYSQL_BUILD=/ssd_root/he923/mysql-server/build
MYSQL_DATA=/ssd_root/he923/mysqldata
```

或者：

```bash
# Release
MYSQL_BUILD=/ssd_root/he923/mysql-server/build-release
MYSQL_DATA=/ssd_root/he923/mysqldata_rls
```

启动前检查已有进程及端口/socket 占用：

```bash
pgrep -a mysqld
ss -ltnp | rg ':33077\b'
ss -lxnp | rg '/tmp/mysql.sock'
```

如果已有实例占用目标数据目录、端口或 socket，先确认并正常关闭对应实例。不要直接删除仍在使用的 socket 或 PID 文件。

启动：

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

这里沿用原脚本的数据路径和 SQL 端口，使用 `--daemonize` 后台运行，并将 TCP 限定为本机、关闭不需要的 MySQL X Protocol。需要远程连接或 X Protocol 时再调整对应参数。内存不足时可将 `8G` 改为 `1G` 或 `2G`。

`--no-defaults` 放在第一个参数，避免读取系统 `my.cnf`；它不会禁止加载数据目录中的 `mysqld-auto.cnf` 持久化变量。`--basedir` 指向构建目录，`--datadir` 指向数据库文件目录，两者不要混淆。

检查启动日志并连接：

```bash
tail -n 80 "$MYSQL_DATA/mysqld.err"

"$MYSQL_BUILD/runtime_output_directory/mysql" \
  --no-defaults --protocol=SOCKET --socket=/tmp/mysql.sock -u root -p
```

在 MySQL 中确认实际连接到哪个实例：

```sql
SELECT VERSION(), @@version_comment, @@basedir, @@datadir, @@port, @@socket;
SELECT 1;
```

尤其要核对 `@@datadir`。`/tmp/mysql.sock` 是几套脚本共用的，能连接不代表连到了预期数据库。

也可以强制 TCP 连接：

```bash
"$MYSQL_BUILD/runtime_output_directory/mysql" \
  --no-defaults --protocol=TCP -h 127.0.0.1 -P 33077 -u root -p
```

`-p` 表示交互输入数据库密码；已有数据库的密码不会因重启改变。若是自行用 `--initialize-insecure` 创建且尚未设置密码的新实例，去掉 `-p`。不要把 Linux 登录密码当作数据库密码。

### 继续使用原脚本

确认没有冲突实例，并接受脚本会停止系统 MySQL 服务后，可二选一：

```bash
bash ~/run_stop_mysql/start_my_mysql.sh       # Debug
# 或者
bash ~/run_stop_mysql/rls_start_my_mysql.sh   # Release
```

这两个脚本会执行 `sudo systemctl stop mysql`，直接删除 `/tmp/mysql.sock` 和各自的 PID 文件，然后用 `&` 启动服务。它们没有等待数据库就绪，所以输出 `Starting custom MySQL...` 不代表启动成功，仍需检查日志和执行 SQL。日常操作优先使用上面的手动命令。

## 3. 正常停止与重启

先通过上一节的 SQL 核对实例，再执行：

```bash
"$MYSQL_BUILD/runtime_output_directory/mysqladmin" \
  --no-defaults --protocol=SOCKET --socket=/tmp/mysql.sock -u root -p shutdown

tail -n 40 "$MYSQL_DATA/mysqld.err"
```

确认日志出现 `Shutdown complete`，对应进程退出后，再重新执行启动命令。

无法通过数据库账号关闭时，可读取所选数据目录的 PID，核实进程后发送 `SIGTERM`：

```bash
MYSQL_PID=$(cat "$MYSQL_DATA/mysqld.pid")
ps -p "$MYSQL_PID" -o pid,user,args
# 核对上面的进程确实对应 MYSQL_DATA 后，再执行：
kill -TERM "$MYSQL_PID"
```

原停止脚本是 `stop_my_mysql.sh` 和 `rls_stop_my_mysql.sh`。它们发送信号后只等待 5 秒，仍未退出就执行 `kill -9`；这可能打断刷盘，使下次启动需要崩溃恢复，因此不作为日常停机首选。正常停止后通常不需要手工删除 PID/socket 文件。

## 4. 编译环境与依赖

本机已确认的环境：

| 项目 | 本机值 / 要求 |
| --- | --- |
| 系统 | Ubuntu 22.04，x86_64 |
| 编译器 | GCC/G++ 11.4.0；本仓库 Linux 配置要求 GCC 至少 11 |
| CMake | `/usr/local/bin/cmake`，3.27.0；当前 FAISS 要求至少 3.24 |
| 生成器 | Unix Makefiles |
| OpenSSL | 系统 OpenSSL 3.0.2 |
| 向量后端 | `WITH_VECINDEX`、`WITH_FAISS`、`WITH_HNSWLIB`、`WITH_DISKANN` 均为 ON |

确认工具链：

```bash
/usr/local/bin/cmake --version
/usr/bin/g++ --version
bison --version
```

当前机器已有依赖。若在另一台同版本 Ubuntu 上准备环境，可按下面的依赖清单安装，再以 CMake 实际检测结果为准：

```bash
sudo apt-get update
sudo apt-get install build-essential bison pkg-config \
  libssl-dev libncurses-dev libaio-dev libnuma-dev \
  libtirpc-dev rpcsvc-proto \
  libboost-dev libboost-program-options-dev \
  libmkl-dev libomp-dev libgoogle-perftools-dev
```

上面没有通过 apt 安装 CMake：先确认可用 CMake 至少为 3.24，本文使用的是本机已安装的 3.27.0。压缩库、Protobuf、ICU 等由源码中的 bundled 版本提供。

FAISS 使用 MKL/BLAS，DiskANN 还需要 Boost program_options、MKL 和 OpenMP。当前 DiskANN 配置会查找 `/usr/lib/x86_64-linux-gnu` 下的 MKL/OMP 库及 `/usr/include/mkl`；非标准安装位置可通过 `-DMKL_PATH=... -DMKL_INCLUDE_PATH=... -DOMP_PATH=...` 指定。

检查外部源码是否齐全：

```bash
cd /ssd_root/he923/mysql-server
git submodule status
ls external/faiss/CMakeLists.txt external/hnswlib/CMakeLists.txt \
   external/DiskANN/mysql/CMakeLists.txt
```

FAISS 和 HNSWLIB 是 Git 子模块。仅在新克隆且子模块尚未初始化时执行 `git submodule update --init --recursive`。当前 FAISS 子模块已有本地修改，不要覆盖或重置。DiskANN 不在根目录 `.gitmodules` 中，迁移源码时还需确保 `external/DiskANN` 实际存在。

## 5. 增量编译：复用已有构建目录

源码改动后，先正常停止需要更新的实例，然后二选一编译：

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

只构建服务端和连接/管理客户端及其依赖，避免把 Router、所有测试和第三方工具一起编译。`8` 是示例并行度，内存不足时降为 `2` 或 `4`。如果确实需要默认全部目标，去掉 `--target mysqld mysql mysqladmin`。

不需要 `make install` 或 `sudo make install`，启动脚本直接使用构建产物：

```text
build[-release]/runtime_output_directory/mysqld
build[-release]/runtime_output_directory/mysql
build[-release]/runtime_output_directory/mysqladmin
build[-release]/library_output_directory/    # 运行依赖的共享库
build[-release]/share/                      # 错误消息等资源
```

不要只拷贝一个 `mysqld` 文件就删除整个构建目录，它还依赖同目录树里的共享库和资源。编译完成后重新启动实例，新进程才会使用新代码；无需重新初始化数据库。

## 6. 首次配置或重建构建目录

现有缓存分别是 `build: Debug + WITH_DEBUG=1` 和 `build-release: Release + WITH_DEBUG=OFF`。需要重新配置时，在源码目录执行下面对应的一组命令。

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

根 CMake 配置会关闭 FAISS 的 GPU、Python 和 C API；本文的 CPU 构建不需要 CUDA。`WITH_CURL=none` 与本机现有缓存一致。

**CPU 指令集注意事项：** 当前 `storage/innobase/vec/CMakeLists.txt` 优先选择存在的 `faiss_avx512` 目标；FAISS 即使设置 `FAISS_OPT_LEVEL=generic` 仍会定义该目标，所以只改这个参数不能保证整个服务端不使用 AVX-512。本机 Xeon Gold 6330 支持 AVX-512；移到其他 CPU 时需要检查/调整目标选择。DiskANN 也启用了 AVX2，并在 Release 下默认使用 `-march=native`。

## 7. 首次初始化：仅用于新的空数据目录

已有的 `mysqldata` 和 `mysqldata_rls` 跳过本节。下面使用独立的新目录，避免触碰已有数据。

```bash
MYSQL_BUILD=/ssd_root/he923/mysql-server/build-release
MYSQL_DATA=/ssd_root/he923/mysqldata_new

# mkdir 不加 -p：目录已存在时不继续初始化。
if mkdir -m 700 "$MYSQL_DATA"; then
  "$MYSQL_BUILD/runtime_output_directory/mysqld" \
    --no-defaults \
    --initialize-insecure \
    --basedir="$MYSQL_BUILD" \
    --datadir="$MYSQL_DATA" \
    --user=he923
else
  printf '目录已存在或创建失败，请先核实，未执行初始化。\n' >&2
fi
```

初始化成功后，保持上面的 `MYSQL_DATA` 变量，按照第 2 节启动，但首次连接去掉 `-p`。`--initialize-insecure` 创建初始空密码的 `root@localhost`，本地连接后可交互执行：

```sql
ALTER USER 'root'@'localhost' IDENTIFIED BY '替换为你自己的密码';
```

以后连接和停止时使用 `-p`。原有启动脚本固定指向旧数据目录，不会自动使用这里的 `mysqldata_new`。

## 8. 常见问题

| 现象 | 检查与处理 |
| --- | --- |
| `Can't connect ... /tmp/mysql.sock` | 查看所选数据目录的 `mysqld.err`；检查进程是否退出、socket 路径是否一致。 |
| 连上了但数据库不对 | 查询 `@@datadir`；几个原脚本共用 socket，必要时明确用 TCP 和目标端口连接。 |
| `Address already in use` / socket 被占用 | 确认已有实例；若要并行运行，数据目录、socket、端口、PID、日志都要独立，X Protocol 也要关闭或单独配置。 |
| `Unable to lock ... ibdata1` | 检查是否有另一个进程使用同一数据目录，不要删除 InnoDB 文件。 |
| `Access denied` | 使用该数据目录原来的数据库账号密码；不要重新初始化来解决登录问题。 |
| CMake 版本太低 | 使用 `/usr/local/bin/cmake` 并确认版本至少 3.24。 |
| 缺少 MKL / Intel OMP / Boost | 检查第 4 节依赖及库路径。 |
| `cc1plus` 被 killed | 查看是否内存不足，降低 `--parallel`。 |
| `Illegal instruction` | 检查 CPU 是否支持当前 FAISS/DiskANN 编译使用的指令集，参见第 6 节。 |
| 找不到共享库或 `errmsg.sys` | 保留完整构建目录，检查 `--basedir`；用 `ldd .../mysqld` 检查 `not found`。 |
| 日志出现 InnoDB assertion / signal 6 | 属于崩溃线索，需要结合最新日志分析。旧 Debug 日志中已有此类记录，但仅凭历史日志不能判定当前数据状态。 |

## 9. 本次核查范围

已读取启动/停止脚本、源码配置和两套构建缓存；确认两套服务端可执行文件分别输出 `9.4.0-debug`、`9.4.0`，Release 运行库解析未发现缺失。两套服务端对本文主要启动参数的 `--validate-config` 检查均通过。

已在独立临时目录 `/tmp/mysql-readme-cmake.eAymDG` 验证第 6 节 Release 配置参数，CMake 配置和生成成功；记录位于该目录的 `configure.log`。文档中的 17 个 Bash 代码块均通过 `bash -n` 语法检查。未执行完整源码编译。

本文整理期间没有启动、停止或重新初始化现有数据库，也没有重新编译覆盖现有二进制。参数检查不等于实际完成启动，也不能证明现有数据库恢复成功；运行时以启动日志及 SQL 查询结果为准。
