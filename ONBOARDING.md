# Onboarding Document

## High-Level Overview

This project provides a lightweight, dependency-free C implementation for running Qwen3-architecture language models. It is designed for simplicity and educational purposes, allowing engineers to understand the inner workings of LLM inference. The key features include:

*   **Minimalism:** The entire inference engine is contained in a single C file (`runq.c`) with no external dependencies.
*   **Cross-Platform:** Supports Linux, macOS, and Windows.
*   **CPU-Optimized:** Runs efficiently on CPUs, with support for multi-core processing via OpenMP.
*   **Quantization:** Includes a Python utility (`export.py`) to convert Hugging Face models to a quantized format (Q8_0), reducing memory and computational requirements.

## Project Structure

This repository contains the following key files:

*   **`runq.c`**: The core of the project, this C file implements the Qwen3 inference engine. It handles model loading, quantization, and the forward pass of the transformer model.
*   **`model.py`**: Defines the Qwen3 transformer architecture in Python using PyTorch. This file is used by `export.py` to load and process Hugging Face models.
*   **`export.py`**: A Python script for converting Hugging Face Qwen3 models into the quantized format used by `runq.c`. It handles model loading, quantization, and serialization of the model weights.
*   **`win.c` and `win.h`**: These files provide a Windows compatibility layer for the C code, ensuring that the project can be built and run on Windows.
*   **`Makefile` and `build_msvc.bat`**: These are the build scripts for the project. The `Makefile` is used for building on Linux and macOS, while `build_msvc.bat` is used for building with MSVC on Windows.
*   **`requirements.txt`**: This file lists the Python dependencies required to run `export.py`.
*   **`README.md`**: The main project README, providing a general overview and instructions for getting started.
*   **`ONBOARDING.md`**: This document, providing a more detailed explanation of the project for new engineers.

## Core Concepts

### Qwen3 Architecture

The project is specifically designed to run models based on the Qwen3 architecture. The core components of this architecture are defined in `model.py` and include:

*   **RMSNorm:** A normalization technique used throughout the model.
*   **Rotary Positional Embeddings (RoPE):** Used to encode positional information.
*   **Grouped Multi-Query Attention:** An efficient attention mechanism.
*   **SwiGLU Activation:** The activation function used in the feed-forward network.

### Quantization (Q8_0)

To run large language models on consumer hardware, this project uses an 8-bit quantization scheme (Q8_0). This process, implemented in `export.py`, involves:

1.  **Grouping:** The model's weights are divided into groups of a fixed size.
2.  **Scaling:** For each group, a scaling factor is calculated based on the maximum absolute value in that group.
3.  **Quantizing:** The weights in each group are then quantized to 8-bit integers.

This process significantly reduces the memory footprint of the model, with a minimal impact on performance.

### C Inference Engine

The C inference engine (`runq.c`) is responsible for loading the quantized model and performing the forward pass. The key steps in this process are:

1.  **Model Loading:** The quantized model weights and metadata are read from the `.bin` file.
2.  **Memory Mapping:** The model data is memory-mapped for efficient access.
3.  **Forward Pass:** For each input token, the engine performs the forward pass of the transformer, including matrix multiplications, normalization, and attention calculations.
4.  **Token Sampling:** The next token is sampled from the output logits using one of several methods (greedy, top-p, etc.).

## How it Works

This section provides a step-by-step walkthrough of the process, from a Hugging Face model to running inference in C.

### 1. Model Conversion and Export

The first step is to convert a pre-trained Hugging Face Qwen3 model into the format required by the C inference engine. This is done using the `export.py` script:

```bash
python export.py <output_file.bin> <hugging_face_model>
```

This script performs the following actions:

1.  **Loads the Hugging Face Model:** It downloads and loads the specified model using the `transformers` library.
2.  **Quantizes the Weights:** It applies the Q8_0 quantization scheme to the model's weights.
3.  **Serializes the Model:** It writes the quantized weights and model metadata to the output `.bin` file.
4.  **Exports the Tokenizer:** It creates a `.tokenizer` file containing the model's vocabulary and merge rules.

### 2. C Inference

Once the model has been exported, you can run inference using the `runq` executable:

```bash
./runq <model_file.bin>
```

The C inference engine (`runq.c`) performs the following steps:

1.  **Loads the Model:** It reads the model metadata and quantized weights from the `.bin` file.
2.  **Initializes the Run State:** It allocates memory for the model's activations, key-value cache, and other runtime data.
3.  **Tokenizes the Input:** It uses the `.tokenizer` file to convert the input prompt into a sequence of tokens.
4.  **Performs the Forward Pass:** For each token in the input sequence, it executes the forward pass of the transformer model.
5.  **Generates Output Tokens:** It samples new tokens from the model's output logits and decodes them back into text.

## Building and Running

### Dependencies

Before building, ensure you have the following dependencies installed:

*   **Python 3:** Required for the `export.py` script.
*   **PyTorch and Transformers:** Install these using `pip install -r requirements.txt`.
*   **C Compiler:** A C compiler such as GCC, Clang, or MSVC is required to build the C inference engine.

### Building on Linux and macOS

The project uses a `Makefile` for building on Linux and macOS. The following are the most common build commands:

*   **Standard Build:**
    ```bash
    make run
    ```
*   **Optimized Build with OpenMP (Recommended):**
    ```bash
    make openmp
    ```

### Building on Windows

For Windows, you can use the provided `build_msvc.bat` script with the Visual Studio Command Prompt:

```bash
build_msvc.bat
```

### Running Inference

After building the project, you can run inference using the `runq` executable:

```bash
./runq <model_file.bin>
```

For more options, such as setting the temperature or using a system prompt, run the executable without any arguments:

```bash
./runq
```

## Deep Dive into `runq.c`

This section provides a more detailed explanation of the C implementation in `runq.c`.

### Core Data Structures

The `runq.c` file defines several key data structures that are used to manage the model and its state:

*   **`Config`**: This struct holds the model's configuration, including the dimensions of the transformer, the number of layers and heads, the vocabulary size, and the maximum sequence length.
*   **`QuantizedTensor`**: This struct represents a quantized tensor, containing a pointer to the quantized 8-bit integer data (`q`) and a pointer to the floating-point scaling factors (`s`).
*   **`TransformerWeights`**: This struct contains all the weights of the model, including the token embeddings, RMSNorm weights, and the quantized weights for the attention mechanism and feed-forward network.
*   **`RunState`**: This struct holds the runtime state of the model, including the activations, key-value cache, and output logits.
*   **`Tokenizer`**: This struct manages the BPE tokenizer, including the vocabulary, merge scores, and token IDs for the beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens.
*   **`Sampler`**: This struct holds the parameters and state for the token sampler, including the vocabulary size, temperature, top-p value, and the random number generator state.

### Neural Network Building Blocks

The `runq.c` file contains several functions that form the building blocks of the transformer model:

*   **`rmsnorm`**: This function implements the RMSNorm algorithm. It takes an input vector, a weight vector, and the size of the vectors as input, and it normalizes the input vector and scales it by the weight vector.
*   **`softmax`**: This function implements the softmax function, which is used to convert a vector of logits into a probability distribution. It takes a vector of floats and its size as input, and it exponentiates and normalizes the vector in place.
*   **`matmul`**: This function performs a matrix multiplication between a quantized weight tensor and a quantized input tensor. It is the most computationally intensive part of the forward pass and is optimized for performance using OpenMP for parallelization.

### The `forward` Function

The `forward` function is the heart of the C inference engine. It takes the current token and its position in the sequence as input and returns the logits for the next token. Here's a breakdown of the steps involved:

1.  **Token Embedding:** The function first retrieves the token embedding from the `token_embedding_table`.
2.  **Attention Mechanism:** For each layer in the transformer, the function performs the following steps:
    *   **RMSNorm:** It applies RMSNorm to the input.
    *   **QKV Calculation:** It calculates the query, key, and value vectors by performing matrix multiplications with the corresponding weight matrices.
    *   **RoPE:** It applies rotary positional embeddings to the query and key vectors.
    *   **Multi-Head Attention:** It calculates the attention scores, applies softmax, and computes the weighted sum of the value vectors.
    *   **Output Projection:** It projects the output of the attention mechanism back into the residual stream.
3.  **Feed-Forward Network:** After the attention mechanism, the function performs the following steps for each layer:
    *   **RMSNorm:** It applies RMSNorm to the output of the attention block.
    *   **SwiGLU:** It applies the SwiGLU activation function.
    *   **Output Projection:** It projects the output of the feed-forward network back into the residual stream.
4.  **Final RMSNorm and Classifier:** After the final layer, the function applies one last RMSNorm and then calculates the output logits by multiplying the final hidden state with the classifier weights.

### Tokenizer Implementation

The `runq.c` file includes a Byte Pair Encoding (BPE) tokenizer that is used to convert between text and tokens.

*   **`decode`**: This function is straightforward. It takes a token ID as input and returns the corresponding string from the vocabulary.
*   **`encode`**: This function is more complex. It takes a string as input and converts it into a sequence of tokens. It does this by iteratively merging the most frequent pairs of adjacent tokens until no more merges are possible.

### Sampler Implementation

The `runq.c` file provides several sampling strategies for generating the next token from the model's output logits:

*   **`sample_argmax`**: This is the simplest sampling method. It selects the token with the highest logit value (i.e., the most likely token).
*   **`sample_mult`**: This function performs multinomial sampling, where the next token is chosen randomly based on the probability distribution of the logits.
*   **`sample_topp`**: This function implements top-p (or nucleus) sampling. It selects the next token from a subset of the vocabulary that have a cumulative probability greater than or equal to a given value `p`. This method can produce more diverse and interesting text than argmax sampling.

### Generation and Chat Loops

The `runq.c` file provides two main modes of operation:

*   **`generate`**: This function generates a sequence of tokens from a given prompt. It takes a prompt as input, encodes it into tokens, and then iteratively calls the `forward` and `sample` functions to generate new tokens until the end-of-sequence token is reached or the maximum sequence length is exceeded.
*   **`chat`**: This function provides an interactive chat mode. It takes a system prompt and a user prompt as input, and it alternates between reading user input and generating a response from the model. It also handles the context window by clearing it when it becomes full.
