# Specifies base image and tag
FROM us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest
WORKDIR /root

# Create virtual env 
# ENV VIRTUAL_ENV=/opt/venv
# RUN python3 -m venv $VIRTUAL_ENV   
# #--system-site-packages
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Installs additional packages
# RUN python3 -m venv /opt/venv --system-site-packages
# COPY requirements.txt .
# RUN . /opt/venv/bin/activate -m && pip install --upgrade pip && pip install -r requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install catboost
# -r requirements.txt

# Copies the data to the docker image
COPY train.csv .

# Copies the trainer code to the docker image.
COPY model.py .

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "-u", "model.py"]