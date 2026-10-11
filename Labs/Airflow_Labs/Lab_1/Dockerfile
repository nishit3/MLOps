# Custom Airflow image with the pipeline's Python dependencies baked in.
# Installing at build time (instead of _PIP_ADDITIONAL_REQUIREMENTS at every
# container start) is what the Airflow docs recommend.
FROM apache/airflow:3.1.5

COPY requirements.txt /requirements.txt
# Pinning apache-airflow stops pip from upgrading/downgrading Airflow itself.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
