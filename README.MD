# Pytorch DDP
DDP stands for Distributed Data Parallel, it is a module that helps you train models across multiple GPUs even on multiple nodes. Unlike DataParallel, which duplicated the model and is slower for multi GPU setups, DDP uses proess-level parallelism and is much faster and scalable. 

How Does DDP work? \
DDP creates one proecss per GPU, where each proess manages its corresponding GPU. If you have 4 GOYs, you will run 4 process. 

2. Data parallelsim:
Each GPU gets a slice of the input data (e.g if the batch size is 64, and you have 4 GPUs, each GPU processes 16 samples)

3. Gradient Synchronization:\
After each GPU computes gradients on its slice of data, DDP syncrhonizes these gradients across all GPUs. and for that it uses (NVIDIA COLLECTIVE COMMUNICATION LIBRARY) nccl for communication between GPUs

4. Scalability:
DDP scales well for training on multiple GPYs or even multiple machines. 

# Basic concepts in DDP
Processes: Each GPU gets a process
Communication Backend: Pytorch uses a backend like ncccls for comms between GPUs
Distributed Initialization: Process must be initialized to 'talk' to each other. 

## Step 2: Basic DDP Code
Refer to [src/ddp.py]

