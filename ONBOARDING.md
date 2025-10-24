# Onboarding Document for qwen3.c

Welcome to the qwen3.c project! This document is designed to help new engineers get up to speed with the codebase quickly.

## 1. Project Overview

qwen3.c is a project that allows you to run inference for frontier models based on the Qwen3 architecture, such as Qwen3-4B or DeepSeek-R1-0528-Qwen3-8B, on your local machine without requiring a GPU. The project is designed to be simple and easy to understand, with the core inference logic contained in a single C file with no external dependencies.

The main goals of this project are:

*   **To be a learning resource:** By providing a minimal, self-contained implementation of LLM inference, qwen3.c helps engineers understand the inner workings of transformer models.
*   **To be a practical tool:** It enables users to run powerful, modern LLMs on standard hardware, making it a great way to experiment with the latest models locally.

The project supports multi-CPU core operation for improved performance, Unicode for multi-language input and output, and a "reasoning mode" for models that support it. It also includes a Python tool for converting HuggingFace models into the custom quantized format used by qwen3.c.

## 2. Codebase Structure

The repository is organized into a few key files. Understanding their roles is the first step to understanding the codebase.

*   **`runq.c`**: This is the heart of the project. It's a single C file that implements the entire inference process for Qwen3-architecture models. It handles loading the quantized model, running the forward pass of the transformer, and generating text. It has no external dependencies, making it highly portable.

*   **`model.py`**: This Python script defines the Qwen3 transformer model using PyTorch. It's used by `export.py` to load the weights from a HuggingFace model before they are quantized and saved to a file.

*   **`export.py`**: This is the script used to convert a Qwen3-architecture model from HuggingFace into the custom `.bin` format used by `runq.c`. It downloads the model, processes the weights, and saves them to a file.

*   **`Makefile`**: This file contains the build instructions for `runq.c`. It provides several build options, including a basic build, a debug build, and an optimized build with OpenMP for multi-core support.

*   **`requirements.txt`**: This file lists the Python dependencies required to run `export.py`.

*   **`win.c` and `win.h`**: These files provide Windows-specific compatibility code, allowing the project to be built and run on Windows.

## 3. How it Works

The project's workflow can be broken down into two main stages: model conversion and inference.

### 3.1. Model Conversion

1.  **Download and Export:** The process begins with the `export.py` script. You provide it with the name of a Qwen3-architecture model from HuggingFace, and it downloads the model's weights.

2.  **Quantization:** The script then uses the PyTorch model defined in `model.py` to process the weights. The weights are quantized to 8-bit integers (Q8_0 quantization) to reduce the model's size and improve performance on CPUs.

3.  **Save to `.bin` file:** The quantized weights, along with the model's configuration, are saved to a custom `.bin` file. This file is what `runq.c` will use for inference.

### 3.2. Inference

1.  **Load the Model:** The `runq.c` program is compiled and run with the path to the `.bin` file as an argument. It memory-maps the file and loads the model's configuration and weights.

2.  **Tokenization:** When you provide a prompt, the program uses its built-in Byte Pair Encoding (BPE) tokenizer to convert the text into a sequence of tokens.

3.  **Forward Pass:** The program then performs the forward pass of the transformer model. For each token in the input sequence, it calculates the logits (the unnormalized probabilities) for the next token. This is where the core matrix multiplications and other operations of the transformer are executed.

4.  **Sampling:** The logits are then passed to a sampler, which selects the next token. The sampling can be done greedily (picking the most likely token) or using more advanced methods like top-p sampling to introduce some randomness.

5.  **Decoding and Output:** The chosen token is then decoded back into text and printed to the console. This process is repeated until the model generates an end-of-sequence token or reaches the maximum sequence length.

## 4. Getting Started

This section will guide you through setting up your development environment and running your first model.

### 4.1. Prerequisites

*   A C compiler (like `gcc` or `clang`)
*   Python 3
*   `pip` for installing Python packages

### 4.2. Setup and Build

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/adriancable/qwen3.c
    cd qwen3.c
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Build the C code:**
    For the best performance, it's recommended to build with OpenMP to enable multi-core support.
    ```bash
    make openmp
    ```
    If your system doesn't support OpenMP, you can create a basic build:
    ```bash
    make
    ```

### 4.3. Running a Model

1.  **Download and convert a model:**
    Let's use the `Qwen/Qwen3-4B` model as an example. The `export.py` script will download the model from HuggingFace and convert it to the `qwen3.c` format.
    ```bash
    python export.py Qwen3-4B.bin Qwen/Qwen3-4B
    ```

2.  **Run inference:**
    You can now run the compiled C program with the converted model.
    ```bash
    ./runq Qwen3-4B.bin
    ```
    You can also try the reasoning mode:
    ```bash
    ./runq Qwen3-4B.bin -r 1
    ```

## 5. Key Concepts

To contribute to this project, it's helpful to have a basic understanding of the following concepts.

### 5.1. Transformer Architecture

The project is based on the transformer architecture, which is the foundation of most modern LLMs. Key components of the transformer model in this codebase include:

*   **Token Embeddings:** The input tokens are mapped to high-dimensional vectors.
*   **RMS Normalization:** A normalization technique used throughout the model.
*   **Multi-Head Attention:** The core mechanism that allows the model to weigh the importance of different tokens in the input sequence.
*   **Feed-Forward Networks:** Fully connected neural networks that are applied after the attention mechanism.
*   **KV Cache:** A cache for the key (K) and value (V) matrices in the attention mechanism, which speeds up inference by avoiding redundant computations.

### 5.2. Quantization

Quantization is the process of reducing the precision of the model's weights. In this project, the weights are converted from 32-bit floating-point numbers to 8-bit integers (Q8_0). This has two main benefits:

*   **Reduced Memory Usage:** The model file becomes significantly smaller, making it easier to load and run on consumer hardware.
*   **Faster Inference:** Integer arithmetic is generally faster than floating-point arithmetic on CPUs, leading to a noticeable performance improvement.

### 5.3. Byte Pair Encoding (BPE)

BPE is a tokenization algorithm used to convert text into a sequence of integers that the model can understand. It works by iteratively merging the most frequent pairs of bytes in the training data. The `runq.c` program has its own BPE tokenizer to process the input prompt.
