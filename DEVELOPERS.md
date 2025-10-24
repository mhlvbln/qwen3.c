# Developer Documentation

This document provides a technical overview of the `qwen3.c` project, intended for developers who want to understand and contribute to the codebase.

## Codebase Structure

The project is organized into the following key files:

- `runq.c`: The core C implementation for model inference.
- `export.py`: A Python script for converting Hugging Face models.
- `Makefile`: The build script for compiling the C code.
- `win.c`, `win.h`: Windows-specific code.
- `model.py`: Defines the model architecture for the export script.

## C Implementation (`runq.c`)

The `runq.c` file contains the core logic for running inference on Qwen3-architecture models. It's written in pure C with no dependencies, making it portable and easy to compile.

### Key Data Structures

- **`Config`**: This struct holds the model's hyperparameters, such as the transformer dimension, number of layers, and vocabulary size.
- **`QuantizedTensor`**: Represents a tensor that has been quantized to 8-bit integers, with associated scaling factors.
- **`TransformerWeights`**: This struct contains all the weights of the model, including the token embeddings, RMSNorm weights, and the weights for the attention and feed-forward network layers. The weights are stored as `QuantizedTensor`s.
- **`RunState`**: This struct holds the activations and buffers needed during the forward pass, such as the key-value cache and the output logits.
- **`Transformer`**: This is the main struct that brings everything together, containing the model's configuration, weights, and run state.
- **`Tokenizer`**: This struct manages the vocabulary and the logic for encoding and decoding text.
- **`Sampler`**: This struct handles the sampling of tokens from the output logits, with support for temperature-based and top-p sampling.

### Core Functions

- **`main`**: The entry point of the program, responsible for parsing command-line arguments, loading the model, and starting the generation or chat loop.
- **`build_transformer`**: This function reads the model checkpoint file, allocates memory for the `Transformer` struct, and memory-maps the model weights.
- **`forward`**: This is the heart of the inference engine. It takes a token and a position in the sequence and performs a forward pass of the transformer, calculating the logits for the next token.
- **`generate`**: This function takes a prompt and generates a sequence of tokens until an end-of-sequence token is produced or the maximum sequence length is reached.
- **`chat`**: This function implements a conversational loop, allowing for interactive chats with the model.
- **`encode`** and **`decode`**: These functions, part of the `Tokenizer`, handle the conversion between text and tokens.
- **`sample`**: This function, part of the `Sampler`, takes the logits from the `forward` pass and samples the next token.

## Python Export Script (`export.py`)

The `export.py` script is a command-line tool for converting pretrained Qwen3-architecture models from Hugging Face into the quantized format required by `runq.c`.

### Key Functions

- **`load_hf_model`**: This function loads a model from the Hugging Face Hub using the `transformers` library. It then extracts the model's weights and configuration and loads them into a custom `Transformer` object, defined in `model.py`.
- **`quantize_q80`**: This function performs Q8_0 quantization on a given tensor. This is a symmetric quantization scheme where the weights are scaled to fit within the range of an 8-bit signed integer (`-127` to `127`). To minimize quantization error, the quantization is done in groups of 64 values.
- **`model_export`**: This is the main function for exporting the model. It takes the `Transformer` object, quantizes the weights using `quantize_q80`, and then serializes the model to a binary file. The file format consists of a 256-byte header followed by the model weights. The header contains the model's configuration, and the weights are a mix of `fp32` (for RMSNorm) and `int8` (for quantized tensors).
- **`build_tokenizer`**: This function creates a custom tokenizer file (`.tokenizer`) from the Hugging Face tokenizer. This file contains the vocabulary and merge scores needed for the BPE tokenizer in `runq.c`.
- **`build_prompts`**: This function generates prompt template files (`.template*`) based on the chat template from the Hugging Face tokenizer. These templates are used in `runq.c` to format prompts for chat mode.

## Build Process

The project uses a simple `Makefile` for compilation.

- **`make`**: This command compiles `runq.c` into an executable named `runq`. It uses `gcc` with the `-O3` optimization level and `-lm` to link the math library.
- **`make openmp`**: This command compiles the code with OpenMP support, which allows the program to use multiple CPU cores for improved performance. The `-fopenmp` flag is added to the `gcc` command.
- **`make clean`**: This command removes the compiled `runq` executable.

For Windows, the `build_msvc.bat` script can be used to compile the code with the MSVC compiler.
