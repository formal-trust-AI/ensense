# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    build-essential \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    

# Copy the current directory contents into the container at /usr/src/app
COPY ./models/ ./models/
# COPY ./models/load_tests ./models/load_tests
COPY ./utils ./utils
COPY ./src ./src
COPY ./outputs/output ./outputs/output
COPY ./requirements.txt .
COPY ./README.md .
COPY ./Dockerfile .
COPY ./images . 
COPY ./unit-tests ./unit-test

# Install any needed packages specified in requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container (Optional, only for web apps)
#EXPOSE 80

# Define environment variable (optional)
#ENV NAME World

# Run app.py when the container launches
# CMD ["python", "./src/sensitive.py"]
# CMD python ./src/sensitive.py $RUN_OPTS
CMD ["/bin/bash"]
