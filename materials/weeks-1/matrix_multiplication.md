# Matrix Multiplication

## Mathematical Foundations

**Definition and Dimensions:** Matrix multiplication is a binary operation that takes two matrices and produces a new matrix. For a product A × B to be defined, the number of columns of A must equal the number of rows of B. If A is of size m × n and B is n × p, then their product C = AB will be an m × p matrix. Each entry of C is obtained by the dot-product of a row of A with a column of B. In other words, for each row i of A and column j of B:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \, B_{kj}$$

where the summation runs over the n common dimension. This formula says that to compute the (i,j) entry of C, you multiply corresponding elements of the i-th row of A and the j-th column of B, then sum those products. The figure below visualizes this process for compatible matrix shapes.

**Figure:** Visualizing matrix multiplication. The number of columns of the first matrix A (here n) must equal the number of rows of the second matrix B (also n). The resulting matrix C = A × B has the same number of rows as A and the same number of columns as B. Each entry $c_{ij}$ is computed by multiplying the entries of the i-th row of A with the entries of the j-th column of B and summing them up.

**Step-by-Step Multiplication:** When multiplying two matrices, systematically multiply each element of a given row in the first matrix by the corresponding element of a given column in the second matrix, and add up those products. For example, let

$$A = \begin{pmatrix}1 & 2 & 3\\4 & 5 & 6\end{pmatrix}, \qquad B = \begin{pmatrix}7 & 8\\9 & 10\\11 & 12\end{pmatrix}$$

Matrix A is 2×3 and B is 3×2, so the product AB will be 2×2. We calculate each entry of C = AB as described:
- $c_{11} = 1 \cdot 7 + 2 \cdot 9 + 3 \cdot 11 = 58$
- $c_{12} = 1 \cdot 8 + 2 \cdot 10 + 3 \cdot 12 = 64$
- $c_{21} = 4 \cdot 7 + 5 \cdot 9 + 6 \cdot 11 = 139$
- $c_{22} = 4 \cdot 8 + 5 \cdot 10 + 6 \cdot 12 = 154$

So, the product is

$$AB = \begin{pmatrix}58 & 64\\139 & 154\end{pmatrix}$$

Notice how each entry of C comes from a row of A and a column of B multiplied element-wise and summed (often called a "dot product"). This procedure can be followed for any matrices (of compatible sizes) to compute their product.

**Key Properties:** Matrix multiplication obeys many of the familiar algebraic properties of ordinary arithmetic, except commutativity. Below are the main properties, given matrices of appropriate dimensions (and scalars r,s):

- **Associative:** A(BC) = (AB)C. In other words, the product of three matrices does not depend on which multiplication is done first. This associativity means we can write ABC without ambiguity in how it is parenthesized.

- **Distributive:** Matrix multiplication distributes over addition. That is, A(B + C) = AB + AC (left-distributive) and (B + C)A = BA + CA (right-distributive). This property is analogous to distribution in scalar algebra, but we must respect the order of factors (note that A(B+C) is not the same expression as (B+C)A unless commutativity holds in special cases).

- **Identity Element:** There are identity matrices (denoted I) that act as multiplicative identities. $I_n$ is the n×n identity matrix (1's on the main diagonal, 0's elsewhere). If A is m × n, then $I_m A = A I_n = A$. Multiplying by an identity matrix leaves any compatible matrix unchanged, on either the left or right side.

- **Non-Commutative:** In general, matrix multiplication is not commutative – the order matters. Usually $AB \neq BA$. Even when both AB and BA are defined and square, they can yield different results or magnitudes. For instance, one can find simple 2×2 matrices where $AB \neq BA$. Only under special conditions (such as when A and B are both diagonal matrices, or one is the identity) will AB = BA. Non-commutativity is a crucial difference from multiplication of real numbers.

- **Transpose of a Product:** The transpose operation reverses the order of multiplication: $(AB)^T = B^T A^T$. In words, the transpose of a product is the product of the transposes in the opposite order. This property can be derived from the definition of matrix product, and it holds for any conformable matrices. For example, if

$$A = \begin{pmatrix}a & b\\ c & d\end{pmatrix} \text{ and } B = \begin{pmatrix}p & q\\ r & s\end{pmatrix}$$

then

$$(AB)^T = \begin{pmatrix}ap + br & aq + bs\\cp + dr & cq + ds\end{pmatrix}^T = \begin{pmatrix}ap + br & cp + dr\\aq + bs & cq + ds\end{pmatrix}$$

and

$$B^T A^T = \begin{pmatrix}p & r\\ q & s\end{pmatrix} \begin{pmatrix}a & c\\ b & d\end{pmatrix} = \begin{pmatrix}ap + br & cp + dr\\aq + bs & cq + ds\end{pmatrix}$$

confirming $(AB)^T = B^T A^T$. (More succinctly, the (i,j)-entry of $(AB)^T$ equals the (j,i)-entry of AB, which in turn equals the sum $\sum_k A_{jk}B_{ki}$, matching the (i,j)-entry of $B^T A^T$.)

- **Other notes:** Matrix multiplication with a zero matrix results in a zero matrix of appropriate size (e.g. A · 0 = 0), and scalar multiplication is compatible with matrix multiplication in the sense that r(AB) = (rA)B = A(rB). These properties, along with those above, mean that the set of n × n matrices forms a ring under addition and multiplication (and even an algebra when combined with scalar multiplication).

**Interpretation:** Another way to view matrix multiplication is as a composition of linear transformations. Each matrix represents a linear map, and multiplying matrices corresponds to composing those maps in sequence. This perspective explains why the order matters (non-commutativity): applying transformation A then B (computing BA on a vector) is generally different from applying B then A (computing AB). It also sheds light on properties like associativity – (AB)C means "do A, then B, then C," which is naturally the same as "do A, then (B then C)." In summary, matrix multiplication provides the algebraic rules for combining linear operations, and these rules closely mirror regular arithmetic with a few important exceptions.

**Example:** To illustrate non-commutativity, consider

$$A = \begin{pmatrix}1 & 1\\ 0 & 0\end{pmatrix} \text{ and } B = \begin{pmatrix}1 & 0\\ 1 & 0\end{pmatrix}$$

Then

$$AB = \begin{pmatrix}2 & 0\\ 0 & 0\end{pmatrix} \text{ but } BA = \begin{pmatrix}1 & 1\\ 1 & 1\end{pmatrix}$$

Clearly $AB \neq BA$. However, if B were the 2×2 identity matrix, then AB = BA = A. As highlighted in lecture, multiplying any matrix by the identity (on either side) returns the original matrix, and if two matrices happen to be diagonal, they commute as well.

## Code Examples: Matrix Multiplication in NumPy and PyTorch

Working with matrices in code is made easy by libraries like NumPy (for general scientific computing in Python) and PyTorch (for tensor computing and deep learning). These libraries implement matrix operations in optimized, low-level code, so you can write math-like operations without explicit loops. Below, we demonstrate basic matrix multiplication and related operations in both NumPy and PyTorch, including examples of broadcasting, batching, and using GPU acceleration.

### Using NumPy (Python)

NumPy represents matrices as numpy.ndarray objects. The simplest way to do matrix multiplication in NumPy is using the @ operator or the np.dot/np.matmul functions. By default, the * operator performs element-wise multiplication, not matrix product, so be sure to use the dedicated matrix multiply operations.

```python
import numpy as np

# Define two matrices A (2x3) and B (3x2)
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7,  8],
              [9,  10],
              [11, 12]])

# Matrix multiplication (2x3) @ (3x2) -> (2x2)
C = A @ B   # or np.dot(A, B) or np.matmul(A, B) produce the same result for 2D arrays
print(C)
# Output:
# [[ 58  64]
#  [139 154]]
```

In this example, A and B are created as 2-dimensional NumPy arrays. The expression A @ B computes their matrix product, yielding the 2×2 result we calculated by hand earlier. NumPy will raise an error if the dimensions are incompatible for multiplication. If you try A * B in this case, it will also error out because NumPy cannot implicitly assume matrix multiply for *. If the shapes were such that element-wise multiplication is possible (same shape arrays), A * B would perform element-wise multiplication, not a dot product. Thus, it's important to use @ or np.dot for true matrix multiplication.

NumPy can also handle higher-dimensional arrays and will by default treat the last two axes of the arrays as matrices when using np.matmul or the @ operator. This allows for batching of matrix operations (more on that shortly).

### Using PyTorch (CPU and GPU Tensors)

PyTorch uses tensors, which generalize matrices to potentially higher dimensions. Basic usage of PyTorch for matrix multiplication is quite similar to NumPy:

```python
import torch

# Define the same matrices as torch tensors
X = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
Y = torch.tensor([[ 7.,  8.],
                  [ 9., 10.],
                  [11., 12.]])

Z = X @ Y  # matrix multiply (2x3 @ 3x2 -> 2x2)
print(Z)
# tensor([[ 58.,  64.],
#         [139., 154.]])
```

Here, X and Y are 2-D tensors (analogous to matrices) of type float. We use the same @ operator to multiply them. PyTorch also provides torch.mm(X, Y) for 2D matrix multiplication and a more general torch.matmul(X, Y) that can handle batching and higher dimensions. In fact, X @ Y is equivalent to torch.matmul(X, Y) in recent PyTorch. One subtle difference is that torch.mm does not broadcast dimensions, whereas torch.matmul supports broadcasting of batch dimensions. This means torch.matmul can be used to multiply, for example, a batch of matrices by another batch in one call, as long as non-matrix dimensions are compatible (either equal or 1 so they can expand).

**Broadcasting example:** Suppose v is a 3-element row vector and M is a 3×3 matrix. In PyTorch, you can do v @ M^T to multiply the vector by each row of M, but if you want to add v to every row of a matrix A (say A is 5×3), you can rely on broadcasting: A + v will automatically broadcast v across the 5 rows (treating v as if it were 5×3 by repeating it) and perform elementwise addition. Similarly, for multiplication, if you have a tensor of shape (batch, m, n) and a matrix of shape (n, p), torch.matmul will broadcast the single matrix across the batch. For example:

```python
# Batch of 10 matrices of shape 3x4, and single matrix of shape 4x5
batch_A = torch.randn(10, 3, 4)
B_single = torch.randn(4, 5)

# Using torch.matmul with broadcasting: B_single is treated as (10,4,5)
batch_C = torch.matmul(batch_A, B_single)
print(batch_C.shape)  # torch.Size([10, 3, 5])
```

In this code, batch_A is a 3D tensor containing 10 separate 3×4 matrices, and B_single is a 4×5 matrix. torch.matmul detects that the first operand has an extra batch dimension of 10, while the second does not, and thus it automatically replicates B_single across those 10 and performs 10 matrix multiplications in parallel. The result batch_C has shape 10×3×5. This broadcasting mechanism is very powerful for performing batched operations without explicit loops.

**Batch matrix multiplication:** Another way to multiply batches of matrices in PyTorch is to use torch.bmm (batch matrix multiply) for 3D tensors. This function requires both inputs to be 3D tensors with the same batch size (no broadcasting), but can be useful for clarity:

```python
# Two batches of matrices (each batch has 10 matrices)
A_batch = torch.randn(10, 3, 4)
B_batch = torch.randn(10, 4, 5)
C_batch = torch.bmm(A_batch, B_batch)
print(C_batch.shape)  # torch.Size([10, 3, 5])
```

This performs 10 independent 3×4 × 4×5 multiplies. In many deep learning scenarios (e.g. processing multiple inputs at once), batching like this is standard.

**Using GPU for acceleration:** One of PyTorch's strengths is the ease of using GPU acceleration. If a GPU is available, you can move tensors and models to the GPU, and PyTorch will utilize high-performance libraries (like cuBLAS) to carry out operations in parallel on the GPU. Moving a tensor to GPU is as simple as:

```python
X_gpu = X.cuda()    # or X.to('cuda')
Y_gpu = Y.cuda()
Z_gpu = X_gpu @ Y_gpu
print(Z_gpu.device)  # prints something like "cuda:0"
```

In the above snippet, calling .cuda() (or .to('cuda')) on a tensor transfers it to the GPU memory. The subsequent matrix multiplication X_gpu @ Y_gpu then executes on the GPU's thousands of parallel threads. The result Z_gpu resides on the GPU as well (note the device printout). Using GPU can dramatically speed up large matrix multiplications. For example, multiplying two 1000×1000 matrices is ~10–100× faster on a modern GPU than on a CPU, thanks to massively parallel computation. (Of course, there is overhead in moving data to the GPU, so one typically keeps data on the GPU during heavy computations to amortize this cost.)

**Summary:** Whether using NumPy or PyTorch, the high-level syntax for matrix multiplication is clean – A @ B – while under the hood these libraries handle all the complexity (looping, vectorization, parallelism). PyTorch extends this to support automatic broadcasting for batch dimensions and seamless CPU/GPU operation. Next, we'll see why these capabilities are so important for data science and deep learning.

## Applications in Data Science and Deep Learning

Matrix multiplication is at the heart of many data science and deep learning algorithms. In linear algebra terms, a matrix represents a linear transformation, and multiplying a matrix by a vector (or another matrix) applies one transformation after another. This makes matrix multiplications indispensable for implementing linear models, neural network layers, and more. Below, we outline several key applications and contexts where matrix multiplication plays a central role:

- **Linear Transformations and Feature Engineering:** In data analysis, applying a linear transformation to data (e.g. rotating, scaling, or projecting points in space) is done via matrix multiply. For instance, if A is an m × n data matrix where each row is an n-dimensional data point, and W is an n × p matrix, then the product AW yields an m × p matrix of transformed data points. Each row of AW is the original data point linearly transformed into a new p-dimensional feature space. Many techniques in statistics and data science, like Principal Component Analysis (PCA), involve eigenvectors or singular vectors, which are found by solving matrix equations and effectively rotating/scaling data via matrix multiplications.

- **Neural Network Forward Pass (Fully Connected Layers):** In a neural network, especially a fully connected (dense) network, each layer's computation is essentially a matrix multiplication. A layer with weight matrix W and bias b computes an output vector z = xW + b for input x (if we treat x as a row vector). Often one writes this as z = W^T x + b if x is a column vector, but the core operation is a dot-product of weights and inputs, repeated for each output neuron – exactly matrix multiplication. If we have a batch of inputs stacked into a matrix X (with shape batch_size × input_dim), and W is input_dim × output_dim, then computing XW in one go produces all of the layer's outputs for the batch (shape batch_size × output_dim). This is vastly more efficient than looping over inputs, and it leverages optimized BLAS routines underneath. In code, this might simply be Z = X @ W in NumPy/PyTorch. The use of matrices to represent network layers makes the implementation concise and computationally efficient. As a concrete example, if X is 1 × 3 and W is 3 × 4, then Z = XW yields a 1 × 4 output, corresponding to 4 neurons receiving inputs from 3 features each.

- **Deep Learning Backpropagation (Gradients):** Not only the forward pass, but also the backward pass (backpropagation) in neural networks relies on matrix multiplication. The gradients of the loss with respect to the weight matrix W in a fully-connected layer can be computed using a matrix outer product of the input vector and the gradient of the output. For a batch of data, if X is (batch × n) and the gradient w.rt. outputs (sometimes called δ) is (batch × p), then the gradient w.rt. weights is given by X^T δ, which is an n × p matrix – again a matrix multiplication. This computation uses the same optimized routines, ensuring training can be done efficiently. Similarly, the gradient with respect to the input of a layer can be computed by multiplying by the transpose of the weight matrix (reflecting the (AB)^T = B^T A^T property). In summary, training updates involve products like (error vectors) × (activation vectors)^T or vice versa. Modern frameworks handle these operations and use autograd to symbolically form the necessary matrix products for gradients.

- **Convolutional Layers as Matrix Multiplication:** Although convolutional neural network (CNN) layers are conceptually different, it's noteworthy that even convolutions can be expressed as matrix multiplications through a technique called im2col, which reshapes patch regions into vectors. Libraries like CuDNN do this under the hood – meaning the GPU is effectively performing giant matrix multiplications to implement convolutions. This highlights that matrix multiply is the workhorse even beyond strictly "linear" layers.

- **Transformers and Large Language Models (LLMs):** Transformers, which power many state-of-the-art language models (like BERT, GPT series, etc.), make heavy use of matrix multiplications. The attention mechanism is a prime example: given matrices Q (queries), K (keys), and V (values), attention is computed by multiplying Q with the transpose of K to get attention scores, then multiplying those scores by V for weighted sums. If Q and K are each of shape (sequence_length × d), then computing QK^T yields a (sequence_length × sequence_length) matrix of pairwise dot-products. This is essentially many dot-products (one for each query-key pair) done via matrix multiplication. After a softmax, that matrix multiplies V (sequence_length × d) to aggregate the values. All these steps are matrix multiplies consuming most of the compute in a transformer. In addition, each transformer block has fully connected layers (often two linear layers in the feed-forward network) that are again large matrix multiplications on the order of (sequence_length × model_dim) times (model_dim × 4*model_dim) and back. In a large language model (LLM) with billions of parameters, virtually all those parameters reside in weight matrices, and generating output involves multiplying very high-dimensional activation vectors by those weight matrices. As an example, the final output layer of a language model might multiply a vector of size in the model's hidden dimension by a matrix of size (embedding dimension × vocabulary size) to produce scores for each word in a 50,000-word vocabulary – a huge matrix multiplication repeated for each token generated.

- **Other ML Applications:** Beyond neural nets, many classic algorithms use matrix multiplication. For instance, in linear regression, normal equation solutions involve products like X^T X and X^T y. In clustering (e.g. k-means), distance computations can be vectorized as matrix operations. Dimensionality reduction (PCA, SVD) internally uses methods that are rich in matrix multiplications. Even reinforcement learning and graphics (transforming coordinates) rely on these operations. The ubiquity of matrix multiplication in computations is why high-performance linear algebra libraries (like BLAS, LAPACK) and GPU-accelerated kernels are so critical in both scientific computing and machine learning.

To summarize, matrix multiplication provides the language of linear models. Whether it's the weight layers in a neural network, the transformations in a transformer model's attention, or the accumulation of statistics in an algorithm, using matrix and tensor operations allows these computations to be expressed cleanly and executed efficiently. This is also why frameworks like TensorFlow and PyTorch focus heavily on optimizing matrix/tensor multiply – it often dominates the runtime of training or inference. A deep learning model might perform millions of these multiplications per second during training, so any improvement or optimization at this level has a huge impact.

## Computational Optimizations and Best Practices

Because matrix multiplication is so central, a lot of effort in scientific computing and machine learning goes into performing it fast and efficiently. A naive implementation of matrix multiply (using three nested loops over indices) has a time complexity of O(n³) for multiplying two n × n matrices. While this is the theoretical complexity, real-world performance depends on constants and hardware utilization. Below, we discuss how modern practice makes matrix multiplication as efficient as possible:

- **Vectorization over Loops:** One fundamental principle is to use vectorized operations instead of explicit Python loops. Vectorization means formulating computations to use high-level array operations that execute internally in optimized C/Fortran (or on the GPU), rather than looping in Python. In deep learning, vectorization is highly preferred because it leverages optimized, parallel computations that run much faster on modern hardware. For example, adding two vectors of length N with a Python for loop would take O(N) Python bytecode steps (very slow), whereas a single call to np.add or simply u + v (where u,v are NumPy arrays) delegates the loop to C code, often using SIMD instructions that add multiple numbers in one CPU cycle. When you write C = A @ B instead of triple-looping over indices, you're harnessing years of optimized linear algebra research. In fact, NumPy's implementation of matrix multiply calls highly optimized BLAS (Basic Linear Algebra Subprograms) routines. These routines are typically written in low-level languages and take advantage of CPU vector instructions, cache-aware algorithms, and even advanced algorithms like Strassen or Coppersmith–Winograd for large matrices. The result is that A @ B in NumPy or PyTorch can be orders of magnitude faster than a pure Python loop, even though mathematically they do the same thing. In summary, vectorization exploits the full compute capability of the hardware, whereas Python-level loops do not.

- **Parallelism and Hardware Acceleration:** Matrix multiplication is highly parallelizable. Each output element c_{ij} can be computed independently (aside from reading the input data). Modern CPUs are multicore and also use vectorized instructions (SIMD) that operate on multiple data points at once. But the real game-changer has been the use of Graphics Processing Units (GPUs) and other accelerators (like TPUs) for matrix math. GPUs consist of thousands of smaller cores that can perform arithmetic operations simultaneously. Libraries like cuBLAS (the CUDA BLAS library) implement matrix multiply on GPUs to leverage this massive parallelism. As a result, a GPU can complete large matrix multiplications much faster by distributing the work across many threads. Deep learning frameworks automatically use these when tensors are on GPU. For example, if you do Z = X_gpu @ Y_gpu as in our PyTorch example, under the hood it launches a highly parallel matrix-kernel on the GPU. This parallelism is one reason why a single high-end GPU can often replace dozens of CPUs for training neural networks. It's also worth noting that even on CPU, BLAS libraries will use multi-threading (if available) to split the operation across cores. Parallel processing and optimized instructions are why vectorized operations are so fast. In short, by writing your code in terms of matrix multiplies (and other tensor ops), you allow the library to utilize all the available hardware concurrency.

- **Memory Access and Cache Optimization:** A lot of the speed in matrix multiplication comes from not just raw FLOPs (floating point operations per second), but from organizing those operations to use memory efficiently. Optimized algorithms will use a technique called blocking or tiling, where submatrices that fit in CPU cache are multiplied to reduce slow memory accesses. This is all done behind the scenes in libraries like ATLAS, OpenBLAS, Intel MKL, etc. The key takeaway for a practitioner is that you should trust library implementations over attempting to manually loop in Python or even naive C – the library writers have implemented many such optimizations (loop unrolling, cache blocking, vectorization) that you'd be hard-pressed to beat in Python.

- **Batching and Amortizing Overhead:** We touched on batching in the context of deep learning – processing multiple inputs at once. This is not only statistically beneficial (for stable gradient estimates) but also computationally advantageous. It's more efficient to perform one matrix multiply on a batch of data than many small multiplies on individual data points. The overhead of launching a GPU kernel or even calling a BLAS routine is significant; doing it once for a large problem is better than doing it 1000 times for many small problems. For example, if you have 1000 vectors of length 512 to multiply by a 512×512 weight matrix, you can stack those vectors into a 1000×512 matrix and do one (1000×512)×(512×512) multiply to get a 1000×512 result, instead of 1000 separate (1×512)×(512×512) multiplies. The latter would have much more function call overhead and wouldn't utilize the hardware as fully for each tiny operation. In PyTorch, this might correspond to using a batch dimension in the tensor and relying on torch.matmul to broadcast or using torch.bmm as shown. Efficient batching can also reduce memory transfers and allow better memory coalescing on GPUs. In summary, try to group operations and use matrix/tensor operations on whole batches of data whenever possible.

- **Memory Management and Transfers:** When using accelerators like GPUs, managing memory is crucial. Data transfer between CPU and GPU is comparatively slow. Therefore, one best practice is to minimize data movement. For example, if you need to perform a series of matrix multiplications on data, it's best to transfer the necessary data to the GPU once (e.g., tensors via .to('cuda')) and then do all multiplications there, rather than shuttling data back and forth for each operation. PyTorch will not implicitly move data between devices; an operation like X_gpu @ Y_cpu is invalid. So it's the programmer's job to ensure operands are on the same device. Also, prefer in-place operations or reuse allocated tensors when possible to reduce memory overhead (but be careful not to overwrite variables needed for gradient computation in PyTorch's autograd). Another tip is to consider data types: using lower precision (float16 or bfloat16) can make multiplications faster (utilizing specialized hardware units like NVIDIA's Tensor Cores) and use less memory, at some cost to numerical precision. Many deep learning training pipelines now use mixed precision to speed up matrix math while maintaining model accuracy.

- **Advanced Algorithms:** The classical matrix multiply algorithm is O(n³), but there are faster algorithms (with lower complexity) for very large matrices, like Strassen's algorithm (O(n^2.8)) or the current theoretical best O(n^2.37). These are usually not used for typical matrix sizes in machine learning because they have large constant factors and are less numerically stable. However, for enormous matrices, hybrid approaches might pick a crossover point to use Strassen or others. In practice, the biggest gains come from hardware utilization rather than lowering Big-O for the sizes we encounter in deep learning (which often are large but not astronomically large in dimension).

- **Profile and Optimize:** If you are multiplying very large matrices or doing many multiplications and performance is critical, it's worth using profilers to see where the bottlenecks are. Tools in NumPy and PyTorch (and their logs) can sometimes show if operations are using optimized paths (e.g., PyTorch may use a specialized kernel if matrix sizes meet certain conditions or if using certain GPU architectures). Ensure that BLAS is properly installed and multi-threaded for CPU. In PyTorch, ensure you call torch.set_num_threads() appropriately if needed, and use torch.cuda.synchronize() around timing code to get accurate GPU timing. These are lower-level considerations, but they can help squeeze out maximum performance for matrix-heavy workloads.

To put it succinctly, matrix multiplication in practice is fast because of vectorized code and parallel hardware. By writing your operations in terms of matrix multiplies (and other tensor ops), you tap into highly optimized implementations. Whether it's NumPy using a carefully tuned BLAS routine on CPU or PyTorch dispatching a GPU kernel, the end result is that extremely large matrix products can be computed in fractions of a second. This is the backbone that makes training multi-billion-parameter models feasible.

Finally, remember that the correctness of matrix multiplication (mathematically) is straightforward, but getting it to run fast has been a significant endeavor of the scientific computing community. Thus, when using these operations, always prefer library functions and high-level constructs. They will almost always beat custom Python code in performance, thanks to decades of collective optimization. Keep your computations in this vectorized form, utilize batch operations, and leverage GPUs when available – these practices will ensure that even the most complex matrix operations become a manageable part of your data science or deep learning pipeline.

