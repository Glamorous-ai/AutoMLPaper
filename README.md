# AutoMLPaper
Github repo to go with the paper "Scalable Drug Property Prediction via Automated Machine Learning"


## Running the code 
There are two ways to run the code.

### 1. As a Docker Container
```bash
$ docker build -t demo .
$ docker run --rm -it -v $PWD:/app demo bash run.sh
```

### 2. Using a Poetry Environment
```bash
$ python3 -m pip install poetry
$ poetry shell
$ poetry install --no-root
$ ./run.sh
```