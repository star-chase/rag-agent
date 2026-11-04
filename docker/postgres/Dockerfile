# pgvector 官方镜像没有中文分词。关键词检索需要 pg_jieba 的 jiebacfg。
FROM pgvector/pgvector:pg16 AS builder

USER root
# 只解压 server-dev 的头文件。直接安装这个包会拖上 LLVM，镜像构建又大又容易中途断开。
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git; \
    cd /tmp; \
    apt-get download postgresql-server-dev-16; \
    dpkg-deb -x postgresql-server-dev-16*.deb /; \
    rm -f postgresql-server-dev-16*.deb; \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/lib/postgresql/16/bin:${PATH}"

# 客户端头文件和 libpq.so 在 libpq-dev 里。这个包很小，不会带上 LLVM。
# CMake 3.25 的 FindPostgreSQL 不认识 PostgreSQL 16，所以下面把三个路径写死。
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libpq-dev; \
    test -f /usr/include/postgresql/libpq-fe.h; \
    test -f /usr/include/postgresql/16/server/postgres.h; \
    test -e /usr/lib/x86_64-linux-gnu/libpq.so; \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/jaiminpan/pg_jieba.git /tmp/pg_jieba \
    && git -C /tmp/pg_jieba submodule update --init --recursive \
    && cmake -S /tmp/pg_jieba -B /tmp/pg_jieba/build \
        -DPostgreSQL_INCLUDE_DIR=/usr/include/postgresql \
        -DPostgreSQL_TYPE_INCLUDE_DIR=/usr/include/postgresql/16/server \
        -DPostgreSQL_LIBRARY=/usr/lib/x86_64-linux-gnu/libpq.so \
        -DCMAKE_CXX_FLAGS="-Wall -std=c++11" \
    && cmake --build /tmp/pg_jieba/build -j"$(nproc)" \
    && cmake --install /tmp/pg_jieba/build \
    && mkdir -p /opt/jieba/lib /opt/jieba/extension /opt/jieba/tsearch \
    && cp /usr/lib/postgresql/16/lib/pg_jieba.so /opt/jieba/lib/ \
    && cp /usr/share/postgresql/16/extension/pg_jieba* /opt/jieba/extension/ \
    && cp /usr/share/postgresql/16/tsearch_data/jieba* /opt/jieba/tsearch/

FROM pgvector/pgvector:pg16

COPY --from=builder /opt/jieba/lib/ /usr/lib/postgresql/16/lib/
COPY --from=builder /opt/jieba/extension/ /usr/share/postgresql/16/extension/
COPY --from=builder /opt/jieba/tsearch/ /usr/share/postgresql/16/tsearch_data/

USER postgres
