FROM pytorch/pytorch@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be AS environment
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VREID_COMPILE=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
COPY requirements-robust-linux.txt requirements-robust-linux.lock.txt /tmp/
RUN python -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements-robust-linux.txt -r /tmp/requirements-robust-linux.lock.txt && python -m pip check
RUN python -m venv --without-pip --system-site-packages /opt/falcon-ann-venv && python -m pip --python /opt/falcon-ann-venv/bin/python install --no-cache-dir --no-deps faiss-cpu==1.15.0
ENV PATH="/opt/falcon-ann-venv/bin:${PATH}"
WORKDIR /app

FROM environment AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VREID_DEVICE=cuda VREID_CORE_DIR=/app VREID_BACKEND=october VREID_SERVICE_DATA=/var/lib/vreid
COPY . /app
RUN python -B scripts/fetch_weights.py && python -B scripts/october_entrypoint.py verify
EXPOSE 8765
ENTRYPOINT ["python","-u","-B","scripts/october_entrypoint.py"]
CMD ["serve","--host","0.0.0.0","--allow-remote"]
