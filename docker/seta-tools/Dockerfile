ARG BASE_IMAGE=radixark/miles:latest
FROM ${BASE_IMAGE}

# 只接收构建脚本准备的三个工具，不接收仓库、runtime 或宿主机配置。
COPY docker /usr/bin/docker
COPY docker-compose /usr/local/lib/docker/cli-plugins/docker-compose
COPY docker-buildx /usr/local/lib/docker/cli-plugins/docker-buildx

ARG HARBOR_PYTHON_VERSION=3.13.16
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN docker --version \
    && docker compose version \
    && docker buildx version \
    && uv python install "${HARBOR_PYTHON_VERSION}" \
    && /opt/python/cpython-3.13-linux-x86_64-gnu/bin/python3.13 --version \
    && python --version

LABEL io.seta.harbor-python="${HARBOR_PYTHON_VERSION}"
# Miles editable 安装、Harbor checkout/uv sync、代理与凭据均由部署阶段提供。
