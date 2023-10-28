# Use an official Python runtime as a base image
FROM continuumio/miniconda3:23.9.0-0

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY minGPT /app

# Copy the conda environment file to the container at /app
COPY conda.yaml /app/conda.yaml

# Create and activate the mingpttutorial conda environment
RUN conda env create -f conda.yaml && \
    echo "source activate mingpttutorial" > ~/.bashrc

# Install module in the Conda environment
RUN /bin/bash -c "source activate mingpttutorial && \
    pip install -e ."

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Set the default command to execute when creating a new container
ENTRYPOINT [ "/bin/bash", "-c", "source activate mingpttutorial && $0 $*" ]

# Run Jupyter Lab when the container launches
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]