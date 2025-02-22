# Catology

Catology is a project that identifies cat breeds based on natural language descriptions. This project is part of an Artificial Intelligence course.

## Prerequisites

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`)

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/catology.git
    cd catology
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Training the Model

Before running the main script, you need to train the model using the train script:

```sh
python train.py
```

### Running main

You need to provide the description of you cat, either by a file or by command-line. Thus, to run the main.py file you need to choose between:

```sh
python main.py --file FILE or
python main.py --text TEXT
```