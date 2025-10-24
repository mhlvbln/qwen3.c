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

### Memory Management

The model checkpoint is loaded using memory mapping (`mmap`), which allows for efficient access to the model's weights without loading the entire file into RAM. The `read_checkpoint` function handles this process. The `Transformer` struct is then initialized, and its weights are mapped to the appropriate locations in the memory-mapped file. The `RunState`, which contains the activations and KV cache, is allocated separately on the heap using `calloc`.

### The Forward Pass (`forward` function)

The `forward` function executes a single step of the transformer. For a given token at a specific position, it calculates the logits for the next token in the sequence. Here's a high-level overview of the process:

1.  **Token Embedding**: The input token is converted into its corresponding embedding vector.
2.  **Transformer Layers**: The input vector is processed by a series of transformer layers. Each layer consists of:
    *   **Attention Mechanism**:
        *   **RMSNorm**: The input is normalized.
        *   **QKV Calculation**: The Query, Key, and Value vectors are computed via matrix multiplication.
        *   **RoPE (Rotary Position Embedding)**: Positional information is added to the Query and Key vectors. The Q and K vectors are also normalized using QK-RMSNorm, a feature of the Qwen3 architecture.
        *   **Scaled Dot-Product Attention**: The attention scores are calculated, and a weighted sum of the Value vectors is produced.
        *   **Output Projection**: The result is projected back to the model's dimension.
    *   **Feed-Forward Network (FFN)**:
        *   **RMSNorm**: The output of the attention mechanism is normalized.
        *   **SwiGLU**: The normalized output is passed through a SwiGLU activation function, which involves three separate linear transformations.
    *   **Residual Connections**: The output of both the attention mechanism and the FFN are added to the input of their respective sub-layers, creating residual connections that help with gradient flow during training (and are a key part of the architecture).
3.  **Final Normalization**: The output of the final transformer layer is passed through one last RMSNorm.
4.  **Classification**: The final normalized output is multiplied by the token embedding matrix (which acts as a classifier) to produce the logits for the next token.

### Tokenization

The tokenizer uses a Byte Pair Encoding (BPE) algorithm. The `Tokenizer` struct, loaded from the `.tokenizer` file, contains the vocabulary and merge scores. The `encode` function takes a string and iteratively merges the most frequent pairs of tokens until no more merges can be performed. The `decode` function performs the reverse operation, converting a token ID back into its string representation.

### Sampling

The `Sampler` struct implements several strategies for selecting the next token from the logits produced by the `forward` pass:

-   **Greedy Sampling**: If the temperature is set to 0, the token with the highest logit (the `argmax`) is always chosen.
-   **Temperature Sampling**: The logits are divided by a temperature value. A higher temperature makes the distribution flatter, increasing the randomness of the output.
-   **Top-p (Nucleus) Sampling**: This method samples from the smallest set of tokens whose cumulative probability exceeds a certain threshold (`p`). This avoids sampling from low-probability tokens, which can often lead to nonsensical output.

## Python Export Script (`export.py`)

The `export.py` script is a command-line tool for converting pretrained Qwen3-architecture models from Hugging Face into the quantized format required by `runq.c`.

### Model Conversion

The `load_hf_model` function orchestrates the conversion process. It uses the `transformers` library to download and load the specified model from the Hugging Face Hub. The script then meticulously transfers the weights and configuration from the Hugging Face model to a custom `Transformer` object (defined in `model.py`), which mirrors the structure of the C implementation. This ensures a seamless transition of the model's architecture and parameters.

### Quantization

The `quantize_q80` function is central to the export process. It implements a Q8_0 symmetric quantization scheme, which significantly reduces the model's memory footprint with a minimal impact on performance. Here's how it works:

1.  **Grouping**: The input tensor is divided into groups of a fixed size (typically 64).
2.  **Scaling Factor**: For each group, the maximum absolute value is determined. A scaling factor is then calculated by dividing this maximum value by 127.
3.  **Quantization**: Each value in the group is divided by the scaling factor and then rounded to the nearest integer, resulting in an 8-bit signed integer.
4.  **Error Calculation**: The function also calculates the maximum quantization error for each group, providing a useful metric for assessing the quality of the quantization.

This group-based quantization strategy is crucial for mitigating the impact of outliers in the weight distribution, which could otherwise lead to significant precision loss.

### Output Files

The `export.py` script generates several files:

-   **`.bin` file**: This is the main model checkpoint file. It has a 256-byte header containing the model's configuration (`Config` struct), followed by the model's weights. The RMSNorm weights are stored in `fp32` format, while the attention and FFN weights are quantized and stored as `int8` values, along with their corresponding `fp32` scaling factors.
-   **`.tokenizer` file**: This file contains the necessary information for the BPE tokenizer in `runq.c`. It includes the vocabulary size, the maximum token length, and for each token, its merge score and UTF-8 representation.
-   **`.template*` files**: These files store the prompt templates used for chat mode. The templates are derived from the chat template of the Hugging Face tokenizer and are used to format the conversation history in a way that the model expects.

## Build Process

The project uses a simple `Makefile` for compilation.

- **`make`**: This command compiles `runq.c` into an executable named `runq`. It uses `gcc` with the `-O3` optimization level and `-lm` to link the math library.
- **`make openmp`**: This command compiles the code with OpenMP support, which allows the program to use multiple CPU cores for improved performance. The `-fopenmp` flag is added to the `gcc` command.
- **`make clean`**: This command removes the compiled `runq` executable.

For Windows, the `build_msvc.bat` script can be used to compile the code with the MSVC compiler.
