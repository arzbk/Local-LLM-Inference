# Replace ChatGPT with a Fully Local System

This code uses ExLlamaV2 in Python to run an open source Large Language Model for chat, automation, and instruction following purposes. In this particular example, I am using a technique to guide the model
to produce text in a given, machine friendly format.

The point of this is to be able to prompt an LLM, and expect that the response will always conform to valid JSON - making the output entirely predictable, and suitable for calling API's, filling forms in 
an app, etc.

To run this, you will need a GPU with 24GB of VRAM or higher, and PyTorch installed with CUDA 11.8 support or higher.
