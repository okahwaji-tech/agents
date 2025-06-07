# 🧮 Vector Spaces and Subspaces


!!! abstract "🔑 Key Concept: Vector Space"
    - A **vector space** $V$ over a field $F$ is a set equipped with two operations:
        - Vector addition
        - Scalar multiplication
    - These operations must satisfy ten fundamental axioms.
    - This abstract structure:
        - Underlies all of linear algebra.
        - Forms the mathematical foundation for machine learning algorithms, neural networks, and modern AI systems.

## 📐 Axioms and Properties of Vector Spaces

!!! note "Definition: Vector Space"
    A **vector space** (or linear space) is a set $V$ equipped with two operations: vector addition and scalar multiplication, satisfying a specific list of axioms. Intuitively, vectors can be added together and scaled by numbers (scalars) while staying in the same set.

Formally, for a vector space $V$ over a field $F$ (e.g., real numbers $\mathbb{R}$), the following properties must hold for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and scalars $a, b \in F$:

### 🔢 The Ten Fundamental Axioms

!!! example "Vector Space Axioms"

    **1. Closure under Addition:**

    $$\mathbf{u} + \mathbf{v} \in V$$

    Adding any two vectors yields another vector in $V$.

    **2. Closure under Scalar Multiplication:**

    $$a \mathbf{v} \in V$$

    Scaling any vector by any scalar yields a vector in $V$.

    **3. Commutativity of Addition:**

    $$\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$$

    **4. Associativity of Addition:**

    $$(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$$

    **5. Additive Identity:**

    There exists a zero vector $\mathbf{0} \in V$ such that:

    $$\mathbf{v} + \mathbf{0} = \mathbf{v} \text{ for all } \mathbf{v} \in V$$

    **6. Additive Inverse:**

    For each $\mathbf{v} \in V$, there is a vector $-\mathbf{v} \in V$ such that:

    $$\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$$

    **7. Multiplicative Identity:**

    $$1 \mathbf{v} = \mathbf{v} \text{ for all } \mathbf{v} \in V$$

    where $1$ is the multiplicative identity in the field $F$.

    **8. Associativity of Scalar Multiplication:**

    $$(ab)\mathbf{v} = a(b\mathbf{v})$$

    **9. Distributivity of Scalar over Vector Addition:**

    $$a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$$

    **10. Distributivity of Vectors over Scalar Addition:**

    $$(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$$

!!! tip "Key Insight"
    These axioms ensure that $V$ has an algebraic structure allowing **linear combinations**. A key consequence of the first two axioms (closure properties) is that no matter how you add or scale vectors in $V$, you remain in $V$.

The middle set of axioms provides the familiar behavior of addition (commutativity, associativity, identity, inverses). The last few axioms govern how scalar multiplication interacts with addition and itself. If a set with two operations satisfies all these axioms, it is a vector space.

!!! warning "Important Note"
    These axioms implicitly require the presence of a zero vector and additive inverses (via the identity and inverse axioms), so every vector space contains a special zero element and each vector's negation.

### 🌟 Examples of Vector Spaces

!!! info "Generality of Vector Spaces"
    The concept of vector space is very general – it is not limited to the geometric arrows in 2D or 3D. Classic examples include:

#### 📊 Common Vector Space Examples

=== "Real Number Line"
    **$\mathbb{R}$ - The Real Number Line**

    All real numbers form a 1-dimensional vector space over $\mathbb{R}$ under standard addition and multiplication.

    - **Zero vector:** $0$
    - **Operations:** Standard arithmetic
    - **Dimension:** 1

=== "Euclidean Space"
    **$\mathbb{R}^n$ - Euclidean Space**

    The set of all $n$-tuples of real numbers (column vectors of length $n$) is an $n$-dimensional vector space.

    **Component-wise operations:**

    $$\text{Addition: } (x_1,y_1,z_1)+(x_2,y_2,z_2)=(x_1+x_2, y_1+y_2, z_1+z_2)$$

    $$\text{Scalar multiplication: } a(x,y,z)=(ax, ay, az)$$

=== "Polynomial Spaces"
    **Polynomial Vector Spaces**

    The set of all polynomials (with real coefficients) of degree $\leq k$ is a vector space.

    - **Vectors:** Polynomials $p(x)$
    - **Addition:** Polynomial addition
    - **Scalars:** Real numbers multiplying polynomials
    - **Example:** Quadratic polynomials $ax^2+bx+c$ form a 3-dimensional vector space

=== "Matrix Spaces"
    **Matrix Vector Spaces**

    The set of all $m\times n$ matrices with entries from a field is a vector space.

    **Example calculation:**

    $$2 \begin{pmatrix}1 & 4\\3 & 5\end{pmatrix} + (-1)\begin{pmatrix}0 & 2\\1 & 1\end{pmatrix}$$

    $$= \begin{pmatrix}2 & 8\\6 & 10\end{pmatrix} + \begin{pmatrix}0 & -2\\-1 & -1\end{pmatrix} = \begin{pmatrix}2 & 6\\5 & 9\end{pmatrix}$$

=== "Function Spaces"
    **Function Vector Spaces**

    The set of all real-valued functions on a domain (e.g., $f: \mathbb{R}\to\mathbb{R}$) is a vector space.

    **Operations:**

    $$\text{Addition: } (f+g)(x)=f(x)+g(x)$$

    $$\text{Scalar multiplication: } (a\cdot f)(x)=a \cdot f(x)$$

!!! tip "Abstract Nature of Vector Spaces"
    These examples highlight how abstract the vector space concept is – as long as the elements and operations obey the axioms, we have a vector space. The geometric 2D/3D vectors are just one instance. This abstraction lets us apply linear algebra to many contexts (polynomials, matrices, functions, etc.), not just physical vectors.

!!! question "Why Are These Properties Important?"
    They guarantee that **linear combinations** make sense in $V$. If $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ are in $V$, any linear combination:

    $$a_1\mathbf{v}_1 + a_2\mathbf{v}_2 + \cdots + a_k\mathbf{v}_k$$

    is also a vector in $V$. This closure under linear combination enables powerful techniques like solving linear equations, defining bases and dimensions, and more.

### 🤖 Vector Spaces in Machine Learning/Deep Learning

!!! example "ML Applications of Vector Spaces"
    Almost all data and parameters in machine learning are represented in vector spaces. This fundamental structure enables the mathematical operations that power modern AI systems.

#### 🔤 Word Embeddings and Semantic Arithmetic

In NLP models, embeddings are vectors in $\mathbb{R}^d$ (e.g., $d = 768$ for BERT). These embedding spaces obey the vector space axioms, allowing meaningful arithmetic on representations.

!!! tip "Famous Word Analogy Example"
    **"King is to Queen as Man is to Woman"**

    In a good word embedding space, you can perform vector arithmetic:

    $$\mathbf{v}(\text{king}) - \mathbf{v}(\text{man}) + \mathbf{v}(\text{woman}) \approx \mathbf{v}(\text{queen})$$

    where $\mathbf{v}(w)$ denotes the embedding vector for word $w$.

The fact that this linear combination of word vectors yields another word vector (specifically, one close to "queen") demonstrates the semantic structure captured in the vector space. This analogical reasoning using addition/subtraction is possible because the embedding vectors live in a high-dimensional vector space that respects linear relationships.

#### 🧠 Deep Learning Frameworks and Vector Operations

!!! info "Framework Implementation"
    In deep learning frameworks like PyTorch or TensorFlow, vectors are typically represented as arrays (tensors) of numbers, and all operations follow linear algebra rules. The vector space axioms are implicitly honored by these frameworks:

    - **Closure under addition:** Adding two tensors of the same shape yields another tensor of that shape
    - **Closure under scalar multiplication:** Multiplying a tensor by a scalar yields another tensor

```python
import torch

# Define three example 3-dimensional vectors (PyTorch tensors)
u = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([-4.0, 5.0, 0.5])
w = torch.tensor([2.0, -3.0, 1.0])

# Verify vector space axioms
print("=== Vector Space Axiom Verification ===")

# Commutativity of addition: u + v == v + u
print(f"Commutativity: {torch.allclose(u + v, v + u)}")

# Additive identity: u + 0 = u
zero_vec = torch.zeros(3)
print(f"Additive identity: {torch.allclose(u + zero_vec, u)}")

# Additive inverse: u + (-u) = 0
print(f"Additive inverse: {torch.allclose(u + (-1)*u, zero_vec)}")

# Associativity of addition: (u+v)+w == u+(v+w)
print(f"Associativity: {torch.allclose((u + v) + w, u + (v + w))}")

# Distributivity: a*(u+v) == a*u + a*v
a = 3.5
print(f"Distributivity: {torch.allclose(a * (u + v), a * u + a * v)}")
```

!!! note "🧠 Neural Network Applications"
    These vector space properties enable linear operations in neural networks:

    - **Layer computation**: $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ (linear combination)
    - **Gradient descent**: Vector addition in parameter space
    - **Backpropagation**: Linear combinations of gradients

    The correctness of these operations relies on vectors living in well-defined vector spaces where addition and scalar multiplication are valid.

## Vector Subspaces: Definitions and Properties

!!! abstract "🔑 Key Concept: Vector Subspace"
    A **subspace** is a subset of a vector space that is itself a vector space under the same operations. Think of it as a "smaller" vector space living inside a larger one.

A subspace is a subset of a vector space that inherits the vector space structure. More precisely, if $(V, +, \cdot)$ is a vector space over a field $F$, then a subset $W \subseteq V$ is called a vector subspace of $V$ if:

1. $W$ is non-empty
2. $W$ is **closed under addition**: $\mathbf{u} + \mathbf{v} \in W$ for all $\mathbf{u}, \mathbf{v} \in W$
3. $W$ is **closed under scalar multiplication**: $a\mathbf{u} \in W$ for all $\mathbf{u} \in W$ and scalars $a \in F$

!!! tip "✅ The Subspace Test"
    **Simplified Test**: A non-empty subset $W \subseteq V$ is a subspace if and only if for any vectors $\mathbf{u}, \mathbf{v} \in W$ and any scalars $a, b$:

    $$a\mathbf{u} + b\mathbf{v} \in W$$

    This single condition encapsulates both closure properties:

    - Setting $a=b=1$: closure under addition
    - Setting $b=0$: closure under scalar multiplication
    - Setting $a=1, b=-1$: guarantees zero vector is in $W$

The beauty of this test is that we don't need to verify all ten vector space axioms – the other properties are automatically inherited from the parent space $V$.

### Key Properties of Subspaces

!!! note "📊 Essential Subspace Properties"
    Any subspace $W$ of vector space $V$ has these fundamental properties:

    **1. Contains Zero Vector**

    - $\mathbf{0} \in W$ (since $\mathbf{0} = 0\mathbf{u}$ for any $\mathbf{u} \in W$)

    **2. Closure Properties**

    - Closed under addition: $\mathbf{u} + \mathbf{v} \in W$ for all $\mathbf{u}, \mathbf{v} \in W$
    - Closed under scalar multiplication: $a\mathbf{u} \in W$ for all $\mathbf{u} \in W$, $a \in F$

    **3. Inherits Vector Space Structure**

    - $W$ is itself a vector space with operations restricted from $V$
    - All ten axioms automatically satisfied

    **4. Intersection vs Union Behavior**

    - ✅ **Intersection**: $U \cap W$ is always a subspace
    - ❌ **Union**: $U \cup W$ is generally NOT a subspace

!!! warning "⚠️ Common Pitfall: Union of Subspaces"
    The union of two subspaces is usually **not** a subspace!

    **Example**: In $\mathbb{R}^2$, let $U$ = x-axis and $W$ = y-axis. Both are subspaces, but:

    - $\mathbf{u} = (1,0) \in U$ and $\mathbf{w} = (0,1) \in W$
    - $\mathbf{u} + \mathbf{w} = (1,1) \notin U \cup W$

    The union fails closure under addition!

### Examples of Subspaces

!!! example "🌟 Geometric Subspaces in $\mathbb{R}^n$"
    Subspaces of $\mathbb{R}^n$ are flat geometric objects passing through the origin:

    **1. Lines Through Origin (1D subspaces)**

    - Example: $W = \{t(1,2,3) : t \in \mathbb{R}\} \subset \mathbb{R}^3$
    - Contains origin: $(0,0,0)$ when $t=0$
    - Closed under operations: $t_1(1,2,3) + t_2(1,2,3) = (t_1+t_2)(1,2,3)$

    **2. Planes Through Origin (2D subspaces)**

    - Example: $W = \{(x,y,0) : x,y \in \mathbb{R}\}$ (the $xy$-plane)
    - Contains origin and closed under linear combinations

!!! example "🔢 Algebraic Subspaces"
    **3. Null Space (Kernel)**

    For homogeneous system $A\mathbf{x} = \mathbf{0}$:

    $$\text{null}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$$

    This is a subspace because: $A(c_1\mathbf{x}_1 + c_2\mathbf{x}_2) = c_1A\mathbf{x}_1 + c_2A\mathbf{x}_2 = \mathbf{0}$

    **4. Column Space**

    For matrix $A$ (size $m \times n$):

    $$\text{col}(A) = \{\mathbf{y} : \mathbf{y} = A\mathbf{x} \text{ for some } \mathbf{x} \in \mathbb{R}^n\}$$

    Set of all linear combinations of columns of $A$

    **5. Polynomial Subspaces**

    - $W = \{p(x) : \deg(p) \leq k\}$ (polynomials of degree at most $k$)
    - Even functions: $\{f : f(-x) = f(x)\}$
    - Odd functions: $\{f : f(-x) = -f(x)\}$

### Subspaces in Machine Learning

!!! example "🤖 ML Applications of Subspaces"
    Subspaces appear throughout machine learning and data science:

    **Principal Component Analysis (PCA)**

    - Find low-dimensional subspace of $\mathbb{R}^n$ capturing most data variance
    - Top $k$ principal components span a $k$-dimensional subspace
    - Dimensionality reduction while preserving structure

    **Deep Learning Applications**

    - **Latent spaces**: Autoencoder feature spaces are vector subspaces
    - **Layer outputs**: Column space of weight matrix $W$ defines possible outputs
    - **Regularization**: Constraining features to subspaces (e.g., sparsity)

    **Model Capacity Analysis**

    - Rank-deficient weight matrices → outputs in lower-dimensional subspace
    - Understanding subspace structure helps analyze model expressiveness

!!! example "🏥 Healthcare Application: Medical Image Analysis"
    In medical imaging, subspaces help with:

    - **Feature extraction**: PCA on medical image patches
    - **Noise reduction**: Project noisy images onto clean subspaces
    - **Anomaly detection**: Healthy tissue spans a subspace; anomalies lie outside
    - **Compression**: Store medical images in lower-dimensional subspaces

!!! tip "💻 Code Example: Why Union Fails"
    Let's demonstrate why the union of subspaces is not a subspace using $\mathbb{R}^2$:

    - $U$ = x-axis: vectors of form $(x,0)$
    - $W$ = y-axis: vectors of form $(0,y)$

    Both are subspaces, but $U \cup W$ is not!

```python
import torch

# Define x-axis subspace U and y-axis subspace W in R^2
def in_U(v):  # check if v is on x-axis
    return abs(v[1]) < 1e-6

def in_W(v):  # check if v is on y-axis
    return abs(v[0]) < 1e-6

u = torch.tensor([1.0, 0.0])  # in U (x-axis)
w = torch.tensor([0.0, 1.0])  # in W (y-axis)
sum_vec = u + w               # = (1,1)

print(f"u in U: {in_U(u)}, w in W: {in_W(w)}")  # True, True
print(f"u+w in U: {in_U(sum_vec)}, u+w in W: {in_W(sum_vec)}")  # False, False
print(f"u+w = {sum_vec} is in neither U nor W!")
```

**Result**: $(1,1)$ is in neither $U$ nor $W$, confirming the union fails closure under addition.

**Key Insight**: The intersection $U \cap W = \{\mathbf{0}\}$ IS a subspace (the trivial subspace), but the union is not!

## Intersections and Sums of Subspaces

!!! abstract "🔑 Key Operations on Subspaces"
    When working with multiple subspaces, two fundamental operations are:

    - **Intersection** ($U \cap W$): Vectors common to both subspaces
    - **Sum** ($U + W$): All possible sums of vectors from each subspace

### Intersection of Subspaces

!!! note "📊 Intersection Properties"
    For subspaces $U, W \subseteq V$:

    $$U \cap W = \{\mathbf{v} : \mathbf{v} \in U \text{ and } \mathbf{v} \in W\}$$

    **Key Properties:**

    - Always contains $\mathbf{0}$ (since $\mathbf{0}$ is in every subspace)
    - Is itself a subspace of $V$
    - Represents the largest subspace contained in both $U$ and $W$
    - If $U \cap W = \{\mathbf{0}\}$, we say they intersect **trivially**

!!! example "🌟 Geometric Example"
    In $\mathbb{R}^3$:

    - $U$ = $xy$-plane: $\{(x,y,0) : x,y \in \mathbb{R}\}$
    - $W$ = $xz$-plane: $\{(x,0,z) : x,z \in \mathbb{R}\}$
    - $U \cap W$ = $x$-axis: $\{(x,0,0) : x \in \mathbb{R}\}$

    The intersection is a 1-dimensional line common to both planes.

### Sum of Subspaces

!!! abstract "🔑 Subspace Sum Definition"
    For subspaces $U, W \subseteq V$, their **sum** is:

    $$U + W = \{ \mathbf{u} + \mathbf{w} : \mathbf{u} \in U, \mathbf{w} \in W \}$$

    This consists of all vectors formed by adding one vector from $U$ and one from $W$.

**Why is $U+W$ a subspace?** Consider vectors $\mathbf{v}_1 = \mathbf{u}_1 + \mathbf{w}_1$ and $\mathbf{v}_2 = \mathbf{u}_2 + \mathbf{w}_2$ in $U+W$:

- **Closure under addition**: $\mathbf{v}_1+\mathbf{v}_2 = (\mathbf{u}_1+\mathbf{u}_2)+(\mathbf{w}_1+\mathbf{w}_2) \in U+W$
- **Closure under scaling**: $a\mathbf{v}_1 = (a\mathbf{u}_1) + (a\mathbf{w}_1) \in U+W$
- **Contains zero**: $\mathbf{0} = \mathbf{0} + \mathbf{0} \in U+W$

!!! note "📊 Geometric Interpretation"
    $U+W$ is the **smallest subspace** containing both $U$ and $W$.

    **Examples in $\mathbb{R}^3$:**

    - $U$ = $x$-axis, $W$ = $y$-axis → $U+W$ = $xy$-plane
    - $U$ = $x$-axis, $W$ = $xy$-plane → $U+W$ = $xy$-plane (since $U \subseteq W$)

    **General Rule**: If $U \subseteq W$, then $U+W = W$

### Generalization

One can define the sum of multiple subspaces $U_1, U_2, \ldots, U_k \subseteq V$ as

$$U_1 + U_2 + \cdots + U_k = \{\mathbf{u}_1 + \mathbf{u}_2 + \cdots + \mathbf{u}_k : \mathbf{u}_i \in U_i \text{ for each } i\}$$

This is also a subspace of $V$. For instance, $U_1+U_2+U_3$ is the set of all sums of one vector from each $U_i$. The same closure argument extends by induction. Often, the notation $\sum_{i=1}^k U_i$ is used for this subspace. In practical terms, if each $U_i$ is spanned by some basis, then $U_1+\cdots+U_k$ is spanned by the union of all those basis vectors.

### Relationship between Dimension, Sum, and Intersection

!!! abstract "🔑 Grassmann's Dimension Formula"
    For finite-dimensional subspaces $U, W \subseteq V$:

    $$\dim(U+W) = \dim(U) + \dim(W) - \dim(U \cap W)$$

    **Intuition**: When combining subspaces, we avoid double-counting their overlap.

!!! example "🧮 Dimension Calculation"
    **Example**: In $\mathbb{R}^3$, let $U$ = $x$-axis and $W$ = $y$-axis

    - $\dim(U) = 1$ (line)
    - $\dim(W) = 1$ (line)
    - $\dim(U \cap W) = 0$ (only origin in common)
    - $\dim(U + W) = 1 + 1 - 0 = 2$ (the $xy$-plane)

    **Special Case**: If $U \cap W = \{\mathbf{0}\}$ (trivial intersection):

    $$\dim(U+W) = \dim(U) + \dim(W)$$

### Application in Machine Learning

!!! example "🤖 ML Applications: Subspace Sums"
    Subspace sums appear throughout machine learning:

    **1. Multimodal Feature Fusion**

    - Visual features subspace $U$ + textual features subspace $W$
    - Combined representation lies in $U + W$
    - Enables rich multimodal understanding

    **2. Residual/Skip Connections**

    - Output of layer $i$: subspace $U_i$
    - Output of layer $j$: subspace $U_j$
    - Skip connection: $\mathbf{h} = \mathbf{u}_i + \mathbf{u}_j \in U_i + U_j$

    **3. Transformer Embeddings**

    - Positional encoding subspace: $U$
    - Token embedding subspace: $W$
    - Combined input: $\mathbf{p} + \mathbf{t} \in U + W$

    **Ideal Case**: $U \cap W = \{\mathbf{0}\}$ allows disentangling position and content information.

!!! example "🏥 Healthcare Application: Multi-Source Medical Data"
    In healthcare AI:

    - **Lab results subspace** + **Imaging features subspace** + **Clinical notes subspace**
    - Combined representation captures comprehensive patient state
    - Each modality contributes unique information to diagnosis

## Direct Sums of Subspaces

!!! abstract "🔑 Key Concept: Direct Sum"
    A **direct sum** is a special case where subspaces combine with no overlap. For subspaces $X, Y \subseteq V$:

    $$V = X \oplus Y$$

    means every vector in $V$ has a **unique** decomposition as $\mathbf{v} = \mathbf{x} + \mathbf{y}$ with $\mathbf{x} \in X, \mathbf{y} \in Y$.

!!! note "📊 Direct Sum Conditions"
    $V = X \oplus Y$ if and only if both conditions hold:

    **1. Spanning Condition**

    - $X + Y = V$ (the sum covers the entire space)
    - Every vector can be written as $\mathbf{v} = \mathbf{x} + \mathbf{y}$

    **2. Uniqueness Condition**

    - The decomposition is unique
    - Equivalently: $X \cap Y = \{\mathbf{0}\}$ (trivial intersection)
    - Only way to write $\mathbf{0} = \mathbf{x} + \mathbf{y}$ is with $\mathbf{x} = \mathbf{y} = \mathbf{0}$

These two conditions together imply that $X \cap Y = \{\mathbf{0}\}$. In fact, it can be shown that $V = X \oplus Y$ if and only if $V = X + Y$ and $X \cap Y = \{0\}$. The "if" part is straightforward: if $X+Y=V$ and they only intersect at $0$, take any $v\in V$; it has at least one decomposition $v=x+y$. If there were another $v=x'+y'$, subtracting gives $0 = (x-x') + (y-y')$ with $x-x' \in X$ and $y-y' \in Y$. By uniqueness, this forces $x-x'=0$ and $y-y'=0$, so $x=x', y=y'$, proving uniqueness. Conversely, if decomposition is unique, in particular $0$ can only decompose as $0+0$, implying no non-zero vector can be in both $X$ and $Y$ (so intersection is $\{0\}$); and certainly $X+Y$ must equal $V$ by the assumption that every vector can be expressed as such a sum.

The notion extends to more than two subspaces: We say $V = U_1 \oplus U_2 \oplus \cdots \oplus U_k$ if (1) $V = U_1 + U_2 + \cdots + U_k$ and (2) the intersection of any subcollection of the $U_i$'s is trivial (equivalently, the uniqueness of representing any vector as a sum of vectors from each $U_i$ holds). In practice, a simple criterion for a direct sum of multiple subspaces is that no nontrivial linear combination of vectors from different subspaces can yield zero except the trivial combination.

When $V$ is a direct sum of subspaces $X$ and $Y$, we sometimes call $X$ and $Y$ complementary subspaces. Each element of $V$ can be split into an $X$-part and a $Y$-part in a unique way. A classic example: in $\mathbb{R}^2$, let $X$ be the $x$-axis and $Y$ be the $y$-axis. Then indeed $\mathbb{R}^2 = X \oplus Y$ because any vector $(a,b)$ can be uniquely written as $(a,0)+(0,b)$ with $(a,0)\in X$ and $(0,b)\in Y$. The intersection $X \cap Y = \{\mathbf{0}\}$ (only the origin lies on both the $x$- and $y$-axes). If we had chosen $Y$ to be another line through the origin that is not the $y$-axis, say $Y$ is the line spanned by $(1,1)$, then $X \cap Y = \{0\}$ still (the $x$-axis and the line $y=x$ intersect only at $(0,0)$). Is $\mathbb{R}^2 = X \oplus Y$ in that case? Yes, because $(1,1)$ is not a multiple of $(1,0)$, so together those two directions span $\mathbb{R}^2$ and any $(a,b)$ has a unique decomposition $(a,b) = (c,0) + (d,d)$ for suitable $c,d$. If instead we chose $Y$ to be the line spanned by $(2,0)$, that would not give a direct sum decomposition with $X$ since $Y$ is actually the same subspace as $X$ (just a different basis vector for it), and $X \cap Y = X$ in that degenerate case (not trivial). There would be infinitely many ways to write a vector on the $x$-axis as $x + y$ with $x\in X, y\in Y$ because those are essentially the same subspace.

### Criterion for Direct Sum

**Criterion (for two subspaces):** $V = X \oplus Y$ if and only if $X+Y=V$ and $X \cap Y = \{0\}$. Often, it's easy to check $X+Y = V$ (span condition) and $X \cap Y = \{0\}$ (independence condition) to conclude a direct sum. In terms of dimensions, if $V = X \oplus Y$ in finite dimensions, then $\dim(V) = \dim(X) + \dim(Y)$ (since none of the dimension of $X$ is "wasted" in overlapping with $Y$). Conversely, if $\dim(X)+\dim(Y)=\dim(X+Y)$, it implies $X \cap Y$ must be trivial, hence $X+Y$ is direct.

### Examples

**Example 1:** Decomposing $\mathbb{R}^n$ into complementary subspaces. Consider $X=\{(x,0,0,\ldots,0)\}$ the $x_1$-axis in $\mathbb{R}^n$, and $Y=\{(0,x_2,x_3,\ldots,x_n)\}$ the subspace of vectors with first coordinate $0$. Then $X \cap Y = \{\mathbf{0}\}$ and $X+Y = \mathbb{R}^n$ (any vector splits into its first-coordinate part plus the rest). So $\mathbb{R}^n = X \oplus Y$. Here $Y = (\text{span of basis }e_2,\ldots,e_n)$ can be viewed as a complement of $X$ in $\mathbb{R}^n$ (and vice versa, $X$ is a complement of $Y$).

**Example 2:** Direct sum in matrix spaces. The space of all $n\times n$ real matrices, denoted $M_{n}(\mathbb{R})$, can be seen as the direct sum of two subspaces: $S =$ the space of symmetric matrices and $A =$ the space of skew-symmetric (anti-symmetric) matrices. Any matrix $M$ can be uniquely written as $M = S + A$ where $S = \frac{1}{2}(M + M^T)$ is symmetric and $A = \frac{1}{2}(M - M^T)$ is skew-symmetric. We have $M_{n}(\mathbb{R}) = S \oplus A$. Indeed, $S \cap A = \{\mathbf{0}\}$ (the only matrix that is both symmetric and skew-symmetric is the zero matrix), and clearly $S + A$ gives all matrices (by the formula above, every matrix is the sum of one symmetric and one skew matrix). This is a powerful decomposition in linear algebra and has practical uses (e.g. any square matrix's even and odd parts under transpose). The uniqueness of the decomposition is evident from the formula – it's essentially the projection onto symmetric vs skew components.

### Direct Sums in Machine Learning

The concept of a direct sum appears in ML primarily in how we design or interpret model architectures. One prominent example is multi-head attention in Transformers. The embedding vector (of dimension $d_{\text{model}}$) is often split into $h$ smaller vectors of dimension $d_k = d_{\text{model}}/h$ for the $h$ attention heads. In implementation, these heads operate on different sections of the embedding. We can think of the model's embedding space as partitioned into $h$ subspaces, each subspace corresponding to one head's portion of the vector (this is a simplification, since in practice they use learned linear projections, but conceptually similar). The multi-head attention mechanism projects the input into multiple smaller subspaces and each head focuses on one subspace. Essentially, the embedding vector gets "segmented" into parts that live in different representational subspaces. If we assume these subspaces intersect only at $\{0\}$ (which is ideal, since we want each head to capture distinct aspects of the representation), then the full embedding can be seen as a direct sum of the head subspaces. After processing, the outputs of each head are concatenated (or added) back together, which corresponds to recombining those subspace components. This idea is why it's said that multi-head attention allows the model to attend to information from different representation subspaces in parallel. In simpler terms, each head handles a different part of the feature space, and together they cover the whole space (direct sum decomposition of the feature space).

Another application: sometimes architectures explicitly split an intermediate vector into parts and process them differently. For example, one could split a hidden state vector into two halves, use one half for one purpose and the other half for another, then combine results. By design, those two halves span complementary subspaces of the state space. This is conceptually a direct sum: the hidden state space $V$ is regarded as $V = V_{\text{part1}} \oplus V_{\text{part2}}$. As long as the network ensures those parts don't interfere (no overlapping information unless intended), analysis can treat them separately. This is akin to how, in an RNN, one might manually partition the state vector for different tasks (though more often the partition emerges from training rather than a hard constraint).

To cement understanding, let's do a brief code example demonstrating a direct sum decomposition in a simple case. Consider $\mathbb{R}^2$ with subspace $X = \text{span}\{(1,1)\}$ and subspace $Y = \text{span}\{(1,-1)\}$. These are two distinct lines through the origin. Their intersection $X \cap Y = \{0\}$ (no nonzero vector lies on both lines). Any vector $(a,b)$ in $\mathbb{R}^2$ can be uniquely written as $(a,b) = x + y$ with $x \in X, y \in Y$. In fact, one can solve for such a decomposition: find scalars $c,d$ such that

$$c(1,1) + d(1,-1) = (a,b)$$

This yields a linear system:

$$(c+d, c-d) = (a, b)$$

so $c+d = a$ and $c-d = b$. Solving gives $c = \frac{a+b}{2}$ and $d = \frac{a-b}{2}$. These are unique, confirming the direct sum. Let's verify with code for a random vector and also see what happens if we choose subspaces that are not direct complements:

```python
import torch
# Define basis vectors for subspaces X and Y in R^2
e1 = torch.tensor([1.0, 1.0])    # basis for X
e2 = torch.tensor([1.0, -1.0])   # basis for Y

# Random vector in R^2
v = torch.randn(2)  # e.g., [a, b]

# Solve for coefficients c,d such that c*e1 + d*e2 = v
A = torch.stack([e1, e2], dim=1)  # form 2x2 matrix with e1,e2 as columns
coeffs = torch.linalg.solve(A, v)  # solve A * [c; d] = v
c, d = coeffs.tolist()
print("v:", v.tolist())
print("c and d:", c, d)

# Reconstruct v from c*e1 + d*e2
reconstructed = c*e1 + d*e2
print("c*e1 + d*e2:", reconstructed.tolist())

# Now try a case where subspaces are not complementary
f1 = torch.tensor([1.0, 0.0])  # basis for U (x-axis)
f2 = torch.tensor([2.0, 0.0])  # basis for W (points on x-axis too, essentially same line)
B = torch.stack([f1, f2], dim=1)
try:
    coeffs = torch.linalg.solve(B, v)  # this will fail as B is singular (f1,f2 not independent)
except RuntimeError as e:
    print("Solving failed for non-complementary subspaces (expected, they're not independent).")
```

In the first part, we form matrix $A = [e_1 \; e_2]$ and solve $A \begin{pmatrix}c\\d\end{pmatrix} = v$. For example, if $v=(2,0)$, the solution will yield $c=1, d=1$ (since $1*(1,1)+1*(1,-1)=(2,0)$). The output will show that the reconstructed vector matches $v$ exactly. In the second part, we attempt the same with $U=\text{span}\{(1,0)\}$ and $W=\text{span}\{(2,0)\}$. Here $W$ is not adding a new dimension (it's the same line as $U$), so $U+W$ is just that line, not all of $\mathbb{R}^2$. The matrix $B$ constructed from $(1,0)$ and $(2,0)$ as columns is rank-deficient (not invertible), and `torch.linalg.solve` will throw an error, indicating no unique solution (indeed, if $v=(2,0)$, there are infinitely many solutions like $v = (2,0)+ (0,0)$ or $v=(1,0)+(1,0)$, etc., and if $v$ had any non-zero second component, no solution at all). This aligns with the failure of direct sum conditions – $U \cap W$ is not $\{0\}$ but $U$ itself in this degenerate case, and $U+W \neq \mathbb{R}^2$.

!!! success "🔑 Key Takeaways: Vector Spaces and Subspaces"
    **Vector Spaces** provide the fundamental algebraic structure for linear algebra and machine learning:

    - **Ten axioms** define the essential properties of addition and scalar multiplication.
    - **Examples everywhere**: $\mathbb{R}^n$, polynomials, matrices, functions, embeddings.
    - **Subspaces** are "smaller" vector spaces living inside larger ones.
    - **Operations on subspaces** include intersection ($\cap$, always a subspace), sum ($+$, always a subspace), direct sum ($\oplus$, sum with unique decomposition), while union ($\cup$) is usually NOT a subspace.
    - **ML Applications**: Feature spaces, embeddings, attention mechanisms, and multimodal fusion demonstrate the power of vector spaces.

    **Machine Learning Impact**:
    Understanding vector spaces enables:

    - **Deep learning**: Neural network operations as linear transformations.
    - **NLP**: Word embeddings and semantic arithmetic.
    - **Computer vision**: Image feature spaces and transformations.
    - **Healthcare AI**: Multi-modal medical data fusion.
    - **Model analysis**: Understanding capacity and expressiveness.

    The direct sum concept is particularly powerful in ML – when feature sets or embedding components are independent (capturing orthogonal information), we can think of representations as direct sums of feature subspaces. This viewpoint helps understand multi-component models like multi-head attention, where each head operates on different representation subspaces in parallel.

