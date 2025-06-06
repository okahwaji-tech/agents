# Vector Spaces and Subspaces

## Axioms and Properties of Vector Spaces

A vector space (or linear space) is a set $V$ equipped with two operations: vector addition and scalar multiplication, satisfying a specific list of axioms. Intuitively, vectors can be added together and scaled by numbers (scalars) while staying in the same set. Formally, for a vector space $V$ over a field $F$ (e.g. real numbers $\mathbb{R}$), the following properties must hold for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and scalars $a, b \in F$:

1. **Closure under Addition:** $\mathbf{u} + \mathbf{v} \in V$. (Adding any two vectors yields another vector in $V$).

2. **Closure under Scalar Multiplication:** $a \mathbf{v} \in V$. (Scaling any vector by any scalar yields a vector in $V$).

3. **Commutativity of Addition:** $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$.

4. **Associativity of Addition:** $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$.

5. **Additive Identity:** There exists a zero vector $\mathbf{0} \in V$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$ for all $\mathbf{v}\in V$.

6. **Additive Inverse:** For each $\mathbf{v} \in V$, there is a vector $-\mathbf{v} \in V$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$.

7. **Multiplicative Identity:** $1 \mathbf{v} = \mathbf{v}$ for all $\mathbf{v} \in V$ (here $1$ is the multiplicative identity in the field $F$).

8. **Associativity of Scalar Multiplication:** $(ab)\mathbf{v} = a(b\mathbf{v})$.

9. **Distributivity of Scalar over Vector Addition:** $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$.

10. **Distributivity of Vectors over Scalar Addition:** $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$.

These axioms ensure that $V$ has an algebraic structure allowing linear combinations. A key consequence of the first two axioms (closure properties) is that no matter how you add or scale vectors in $V$, you remain in $V$. The middle set of axioms provides the familiar behavior of addition (commutativity, associativity, identity, inverses). The last few axioms govern how scalar multiplication interacts with addition and itself. If a set with two operations satisfies all these axioms, it is a vector space. Notably, these axioms implicitly require the presence of a zero vector and additive inverses (via the identity and inverse axioms), so every vector space contains a special zero element and each vector's negation.

### Examples of Vector Spaces

The concept of vector space is very general – it is not limited to the geometric arrows in 2D or 3D. Classic examples include:

1. **Real Number Line $\mathbb{R}$:** All real numbers form a 1-dimensional vector space over $\mathbb{R}$ under standard addition and multiplication. The zero vector is $0$.

2. **Euclidean Space $\mathbb{R}^n$:** The set of all $n$-tuples of real numbers (column vectors of length $n$) is an $n$-dimensional vector space. Vector addition and scalar multiplication are done component-wise. For example, in $\mathbb{R}^3$, $(x_1,y_1,z_1)+(x_2,y_2,z_2)=(x_1+x_2, y_1+y_2, z_1+z_2)$ and $a(x,y,z)=(ax, ay, az)$.

3. **Polynomial Spaces:** The set of all polynomials (with real coefficients) of degree $\leq k$ is a vector space. Vectors are polynomials $p(x)$, addition is polynomial addition, and scalars are real numbers multiplying polynomials. For example, the set of all quadratic polynomials $ax^2+bx+c$ is a vector space (of dimension 3).

4. **Matrix Spaces:** The set of all $m\times n$ matrices with entries from a field is a vector space. Two matrices can be added and multiplied by scalars (each entry scaled) to produce another matrix. For instance, 

$$2 \begin{pmatrix}1 & 4\\3 & 5\end{pmatrix} + (-1)\begin{pmatrix}0 & 2\\1 & 1\end{pmatrix} = \begin{pmatrix}2 & 8\\6 & 10\end{pmatrix} + \begin{pmatrix}0 & -2\\-1 & -1\end{pmatrix} = \begin{pmatrix}2 & 6\\5 & 9\end{pmatrix}$$

which is still a matrix of the same size.

5. **Function Spaces:** The set of all real-valued functions on a domain (say all functions $f: \mathbb{R}\to\mathbb{R}$) is a vector space. Here vectors are functions; addition is defined as $(f+g)(x)=f(x)+g(x)$ and scalar multiplication as $(a\cdot f)(x)=a \cdot f(x)$. Many specific function spaces (e.g. spaces of continuous or differentiable functions) are also vector spaces.

These examples highlight how abstract the vector space concept is – as long as the elements and operations obey the axioms, we have a vector space. The geometric 2D/3D vectors are just one instance. This abstraction lets us apply linear algebra to many contexts (polynomials, matrices, functions, etc.), not just physical vectors.

**Why are these properties important?** They guarantee that linear combinations make sense in $V$. If $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ are in $V$, any linear combination $a_1\mathbf{v}_1 + a_2\mathbf{v}_2 + \cdots + a_k\mathbf{v}_k$ is also a vector in $V$. This closure under linear combination enables powerful techniques like solving linear equations, defining bases and dimensions, and more. Many theorems (e.g. relating to spans, linear independence, etc.) hinge on these axioms.

### Vector Spaces in Machine Learning/Deep Learning

Almost all data and parameters in machine learning are represented in vector spaces. For example, an embedding in an NLP model is a vector in $\mathbb{R}^d$ (for some dimension $d$, e.g. 768 for BERT). These embedding spaces obey the vector space axioms, allowing meaningful arithmetic on representations. A famous example is the word analogy: "king is to queen as man is to woman." In a good word embedding space, you can do the vector arithmetic $\mathbf{v}(\text{king}) - \mathbf{v}(\text{man}) + \mathbf{v}(\text{woman}) \approx \mathbf{v}(\text{queen})$. Here $\mathbf{v}(w)$ denotes the embedding vector for word $w$. The fact this linear combination of word vectors yields another word vector (specifically, one close to "queen") is a testament to the semantic structure captured in the vector space. This analogical reasoning using addition/subtraction is possible because the embedding vectors live in a high-dimensional vector space that respects linear relationships.

In deep learning frameworks like PyTorch or TensorFlow, vectors are typically represented as arrays (tensors) of numbers, and all operations (like addition, scaling, dot-products) follow linear algebra rules. The vector space axioms are implicitly honored by these frameworks. For instance, adding two tensors of the same shape yields another tensor of that shape (closure under addition), and multiplying a tensor by a scalar (a Python float/NumPy scalar) yields another tensor (closure under scalar multiplication). Below is a short code snippet illustrating some axioms with PyTorch:

```python
import torch

# Define three example 3-dimensional vectors (PyTorch tensors)
u = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([-4.0, 5.0, 0.5])
w = torch.tensor([2.0, -3.0, 1.0])

# Commutativity of addition: u + v == v + u
lhs = u + v
rhs = v + u
print(torch.allclose(lhs, rhs))  # should output: True

# Additive identity: u + 0 = u
zero_vec = torch.zeros(3)
print(torch.allclose(u + zero_vec, u))  # True

# Additive inverse: u + (-u) = 0
print(torch.allclose(u + (-1)*u, zero_vec))  # True

# Associativity of addition: (u+v)+w == u+(v+w)
lhs = (u + v) + w
rhs = u + (v + w)
print(torch.allclose(lhs, rhs))  # True

# Distributivity: a*(u+v) == a*u + a*v
a = 3.5
lhs = a * (u + v)
rhs = a * u + a * v
print(torch.allclose(lhs, rhs))  # True
```

This code confirms some vector space properties (commutativity, identity, inverse, associativity, distributivity) with concrete numeric examples. In practice, these properties enable linear operations in neural networks: for example, the output of a layer is often computed as $W\mathbf{x} + \mathbf{b}$ (a linear combination of input $\mathbf{x}$ plus a bias vector). The correctness of such operations relies on the fact that $\mathbf{x}$ and $\mathbf{b}$ live in the same vector space so addition is valid, and $W\mathbf{x}$ (a matrix times a vector) produces another vector in that space.

## Vector Subspaces: Definitions and Properties

A subspace is a subset of a vector space that is itself a vector space (under the same operations). More precisely, if $(V, +, \cdot)$ is a vector space over a field $F$, then a subset $W \subseteq V$ is called a vector subspace of $V$ if $W$ is non-empty and for every $\mathbf{u}, \mathbf{v} \in W$ and scalar $a \in F$:

1. The sum $\mathbf{u} + \mathbf{v}$ is in $W$ (closure under addition in $W$), and
2. The scalar multiple $a\mathbf{u}$ is in $W$ (closure under scalar multiplication in $W$).

Equivalently, $W$ must contain the zero vector and be closed under addition and scaling. In short, $W$ is a subspace of $V$ if $W$ itself satisfies all the vector space axioms using the same operations as $V$. We do not need to check all ten axioms for $W$ from scratch; it's enough to ensure the subset is closed under addition and scalar multiplication (and contains $0$), because then all other axioms are automatically inherited from $V$. This criterion is often known as the **Subspace Test**:

**Subspace Test:** A non-empty subset $W \subseteq V$ is a subspace of $V$ if and only if for any vectors $\mathbf{u}, \mathbf{v} \in W$ and any scalars $a, b$, the linear combination $a\mathbf{u} + b\mathbf{v}$ is also in $W$.

This test encapsulates the closure properties (taking $a=b=1$ gives closure under addition, and $b=0$ gives closure under scalar multiplication, and it automatically yields the zero vector when $a=1, b=-1$ or simply $a=b=0$).

### Key Properties of Subspaces

Key properties of any subspace $W$ of $V$:

1. $W$ contains the zero vector of $V$. (Indeed, $0 = 0\mathbf{u} \in W$ for any $\mathbf{u}\in W$ since $W$ is non-empty.)

2. $W$ is closed under addition and scalar multiplication. That is, combining or scaling vectors in $W$ cannot produce a vector outside $W$.

3. Every subspace is itself a vector space (with the same addition and scalar operations restricted to $W$). All the axioms hold in $W$ because they hold in $V$ and $W$ is closed under the operations.

4. **Intersection and Union:** The intersection of any collection of subspaces is also a subspace (common elements satisfy the subspace criteria). In contrast, the union of two subspaces is generally not a subspace unless one subspace is contained in the other. For example, if $U$ and $W$ are two distinct subspaces of $V$, $U \cup W$ usually fails closure (take $\mathbf{u}\in U$, $\mathbf{w}\in W$; unless one is in the other, $\mathbf{u}+\mathbf{w}$ will lie outside the union). This is a common pitfall: two lines through the origin in $\mathbb{R}^2$, each a subspace, have a union that is not a subspace (except if one line equals the other).

### Examples of Subspaces

Subspaces appear in many forms. Intuitively, subspaces of $\mathbb{R}^n$ are flat geometric objects through the origin (lines, planes, etc. through the origin). Here are some examples:

1. **In $\mathbb{R}^3$, any line through the origin** is a one-dimensional subspace of $\mathbb{R}^3$. For instance, $W = \{ t(1,2,3) : t \in \mathbb{R}\}$ is a subspace of $\mathbb{R}^3$ (a line through $(1,2,3)$). It contains $0=(0,0,0)$ (when $t=0$) and is closed under addition/scaling (adding two multiples of $(1,2,3)$ yields another multiple of $(1,2,3)$).

2. **In $\mathbb{R}^3$, any plane through the origin** is a two-dimensional subspace. For example, $W = \{(x,y,0) : x,y \in \mathbb{R}\}$ (the $xy$-plane) is a subspace of $\mathbb{R}^3$. It contains $(0,0,0)$, and adding or scaling vectors that have zero $z$-component keeps the $z$-component zero.

3. **Solution sets of homogeneous linear equations:** If $A\mathbf{x}=\mathbf{0}$ is a homogeneous linear system, the set of all solutions $\{\mathbf{x}: A\mathbf{x}=\mathbf{0}\}$ is a subspace of $\mathbb{R}^n$ (where $n$ is the number of columns of $A$). This set is called the null space (or kernel) of $A$. It is a subspace because $A(c_1\mathbf{x}_1 + c_2\mathbf{x}_2) = c_1A\mathbf{x}_1 + c_2A\mathbf{x}_2 = \mathbf{0}$ for any two solutions $\mathbf{x}_1,\mathbf{x}_2$ and scalars $c_1,c_2$, so any linear combination of solutions is still a solution.

4. **Column spaces and Row spaces:** The set of all linear combinations of the columns of a matrix $A$ (the column space of $A$) is a subspace of $\mathbb{R}^m$ (if $A$ is $m\times n$). Similarly, the row space (all linear combinations of row vectors) is a subspace of $\mathbb{R}^n$. These subspaces are fundamental in linear algebra (relating to the rank of $A$).

5. **Polynomial subspaces:** Consider $V$ = all polynomials, and let $W$ = all polynomials of degree $\leq k$. $W$ is a subspace of $V$. It's closed under addition and scaling (adding two polynomials of degree at most $k$ yields at most degree $k$, etc.). Another example: the set of all even functions (or odd functions) is a subspace of the vector space of all real functions, since the sum of even functions is even, etc.

### Subspaces in Machine Learning

In ML and data science, subspaces often correspond to certain feature subspaces or latent spaces. For example, in PCA (Principal Component Analysis), we find a low-dimensional subspace of $\mathbb{R}^n$ that captures most variance of the data. The top $k$ principal components span a $k$-dimensional subspace (the principal subspace); projecting data onto this subspace reduces dimensionality while preserving key structure. In deep learning, the concept of a latent space (e.g. the space of encoded features in an autoencoder) is essentially a vector space, and sometimes we constrain it to a subspace for regularization (e.g. requiring certain features to be zero – effectively confining data to a subspace). The column space example above has a direct analog: the column space of a network layer's weight matrix is the subspace of outputs that the layer can produce (since any output is $W\mathbf{x}$ for some input $\mathbf{x}$, and thus lies in the span of the columns of $W$). Understanding subspaces can help in analyzing model capacity: for instance, if a model's weight matrices have rank deficiency, the outputs lie in a lower-dimensional subspace of the target space, potentially limiting expressiveness.

To illustrate a simple subspace situation in code, consider $\mathbb{R}^2$ and two subspaces: $U =$ x-axis (all vectors of the form $(x,0)$) and $W =$ y-axis (all vectors of the form $(0,y)$). Both $U$ and $W$ are subspaces of $\mathbb{R}^2$. However, their union $U \cup W$ (all vectors that lie on either the x-axis or y-axis) is not a subspace because it's not closed under addition. For example, $\mathbf{u}=(1,0)\in U$ and $\mathbf{w}=(0,1)\in W$, but $\mathbf{u}+\mathbf{w}=(1,1)$ is not in $U \cup W$ (it's not on either axis). We can check this with a short snippet:

```python
# Define x-axis subspace U and y-axis subspace W in R^2
def in_U(v):  # check if v is on x-axis
    return v[1] == 0

def in_W(v):  # check if v is on y-axis
    return v[0] == 0

u = torch.tensor([1.0, 0.0])
w = torch.tensor([0.0, 1.0])
sum_vec = u + w

print(in_U(u) and in_U(w))      # True, True (u and w individually lie in U or W respectively)
print(in_U(sum_vec), in_W(sum_vec))  # False, False -> sum_vec = (1,1) is in neither U nor W
```

This prints True, True for the first line (meaning $u \in U$ and $w \in W$ individually) and False, False for the second (meaning $(1,1)$ is in neither $U$ nor $W$), confirming the union is not closed under addition. In contrast, the intersection $U \cap W$ in this case is $\{\mathbf{0}\}$, which is a subspace (the trivial subspace). This simple exercise shows the importance of the closure property in defining subspaces.

## Intersections and Sums of Subspaces

When dealing with multiple subspaces, two fundamental operations are their intersection and their sum. We touched on intersections above; here we formalize and explore these concepts further.

### Intersection of Subspaces

If $U$ and $W$ are subspaces of a vector space $V$, then $U \cap W = \{\mathbf{v} : \mathbf{v} \in U \text{ and } \mathbf{v} \in W\}$ is also a subspace of $V$. The intersection is at least $\{\mathbf{0}\}$ (since $0$ lies in every subspace) and possibly larger if $U$ and $W$ have other vectors in common. The intersection $U \cap W$ satisfies the subspace test: if $\mathbf{x}, \mathbf{y}$ are in both $U$ and $W$, then $\mathbf{x}+\mathbf{y}$ is in both $U$ and $W$ (since each is a subspace), hence in the intersection; similarly for scalar multiples. In general, the intersection $U \cap W$ is the largest subspace contained in both $U$ and $W$. For example, in $\mathbb{R}^3$, if $U$ is the $xy$-plane and $W$ is the $xz$-plane, then $U \cap W$ is the $x$-axis (their line of intersection), which indeed is a subspace (1-dimensional line) common to both. If two subspaces have only the zero vector in common (their intersection is $\{0\}$), we say they intersect trivially. This situation is especially important for direct sums (discussed in the next section).

### Sum of Subspaces

Given two subspaces $U, W \subseteq V$, their sum is defined as

$$U + W = \{ \mathbf{u} + \mathbf{w} : \mathbf{u} \in U, \mathbf{w} \in W \}$$

In words, $U+W$ consists of all vectors you can form by adding one vector from $U$ and one from $W$. It's not a disjoint union, but rather all possible sums of one element from each. $U+W$ is itself a subspace of $V$. To see this, consider any two vectors in $U+W$: say $\mathbf{v}_1 = \mathbf{u}_1 + \mathbf{w}_1$ and $\mathbf{v}_2 = \mathbf{u}_2 + \mathbf{w}_2$ with $\mathbf{u}_i \in U, \mathbf{w}_i \in W$. Their sum is $\mathbf{v}_1+\mathbf{v}_2 = (\mathbf{u}_1+\mathbf{u}_2)+(\mathbf{w}_1+\mathbf{w}_2)$. Since $U$ and $W$ are subspaces, $\mathbf{u}_1+\mathbf{u}_2 \in U$ and $\mathbf{w}_1+\mathbf{w}_2 \in W$. Thus $\mathbf{v}_1+\mathbf{v}_2$ is still of the form (something in $U$) + (something in $W$), hence in $U+W$. Similarly, for a scalar $a$, $a\mathbf{v}_1 = (a\mathbf{u}_1) + (a\mathbf{w}_1) \in U+W$. The zero vector is $0+0 \in U+W$. So $U+W$ satisfies the subspace criteria.

Geometrically, $U+W$ can be thought of as the smallest subspace of $V$ that contains both $U$ and $W$. Indeed, any subspace containing $U$ and $W$ must contain all their sums and scalar multiples, so it must contain $U+W$. For example, in $\mathbb{R}^3$, if $U$ is a line along the $x$-axis and $W$ is a line along the $y$-axis, then $U+W$ is the $xy$-plane (all vectors of the form $(a,b,0)$). That plane is the smallest subspace containing both lines. If instead $W$ were the $xy$-plane and $U$ the $x$-axis (which lies within $W$), then $U+W = W$ (summing doesn't go beyond the larger subspace in that case). More generally, if $U \subseteq W$, then $U+W = W$. At the other extreme, if $U$ and $W$ share only the zero vector (no other overlap), the dimension of $U+W$ is $\dim(U) + \dim(W)$ (for finite dimensions), since their sum basically puts the subspaces "together" as direct summands (we formalize this as the direct sum soon).

### Generalization

One can define the sum of multiple subspaces $U_1, U_2, \ldots, U_k \subseteq V$ as

$$U_1 + U_2 + \cdots + U_k = \{\mathbf{u}_1 + \mathbf{u}_2 + \cdots + \mathbf{u}_k : \mathbf{u}_i \in U_i \text{ for each } i\}$$

This is also a subspace of $V$. For instance, $U_1+U_2+U_3$ is the set of all sums of one vector from each $U_i$. The same closure argument extends by induction. Often, the notation $\sum_{i=1}^k U_i$ is used for this subspace. In practical terms, if each $U_i$ is spanned by some basis, then $U_1+\cdots+U_k$ is spanned by the union of all those basis vectors.

### Relationship between Dimension, Sum, and Intersection

For finite-dimensional spaces, there is an important formula called **Grassmann's dimension formula**:

$$\dim(U+W) = \dim(U) + \dim(W) - \dim(U \cap W)$$

This formula reflects that if $U$ and $W$ overlap (non-trivial intersection), we count that overlap once when summing dimensions. If $U$ and $W$ only intersect trivially ($U \cap W = \{0\}$), then $\dim(U+W) = \dim(U)+\dim(W)$. In the previous example, $\dim(x\text{-axis})=1$, $\dim(y\text{-axis})=1$, and $\dim$(their intersection$)=0$, so $\dim(xy\text{-plane}) = 1+1-0 = 2$, which checks out.

### Application in Machine Learning

While we rarely speak explicitly about "sums of subspaces" in applied ML, the concept appears when combining feature sets or learned representations. For instance, consider a scenario with multiple sets of features (say visual features and textual features for an image captioning system). One approach to combine them is to concatenate feature vectors (which corresponds to a direct product of vector spaces), but another approach is to project them into a common vector space and then add them. When we add two feature vectors (one from subspace $U$ of visual features, one from subspace $W$ of textual features), the result lies in $U+W$. In neural networks, skip connections or residual connections effectively add vectors coming from different subspaces (e.g. the output of a previous layer with the output of a deeper layer) – the result lives in the sum of those subspaces. If those subspaces carry complementary information, the sum has richer representation power. In attention mechanisms, it's common to sum positional encodings and word embeddings. Suppose $U$ is the subspace spanned by positional encoding vectors and $W$ is the subspace spanned by token embedding vectors. Individual position or token embeddings lie in $U$ or $W$ respectively, and the combined input embedding is a sum $\mathbf{p}+\mathbf{t}$ with $\mathbf{p}\in U$, $\mathbf{t}\in W$. This sum resides in $U+W$. If $U$ and $W$ overlap minimally (ideally only at $\{0\}$), the model can disentangle content and positional information. In practice, learned embeddings may not be strictly confined to separate subspaces, but the idea of combining different representation subspaces via addition is powerful (and designers try to ensure one doesn't dominate or distort the other, e.g. by scaling).

## Direct Sums of Subspaces

The direct sum is a special case of the sum of subspaces where the intersection is trivial. Suppose $X$ and $Y$ are subspaces of $V$. We say $V$ is the direct sum of $X$ and $Y$ if every vector in $V$ can be expressed uniquely as the sum of a vector from $X$ and a vector from $Y$. In notation, we write:

$$V = X \oplus Y$$

This definition encompasses two conditions:

1. **Spanning (Sum) Condition:** Every $v \in V$ can be written as $v = x + y$ with $x \in X$ and $y \in Y$. (So $X+Y = V$ in terms of set equality, i.e. the sum of the subspaces covers the whole space $V$.)

2. **Uniqueness Condition:** This representation is unique; no vector in $V$ has two different such $X+Y$ decompositions. Equivalently, the only way to write $0$ as $x+y$ with $x \in X, y \in Y$ is the trivial way $x=0, y=0$.

These two conditions together imply that $X \cap Y = \{\mathbf{0}\}$. In fact, it can be shown that $V = X \oplus Y$ if and only if $V = X + Y$ and $X \cap Y = \{0\}$. The "if" part is straightforward: if $X+Y=V$ and they only intersect at $0$, take any $v\in V$; it has at least one decomposition $v=x+y$. If there were another $v=x'+y'$, subtracting gives $0 = (x-x') + (y-y')$ with $x-x' \in X$ and $y-y' \in Y$. By uniqueness, this forces $x-x'=0$ and $y-y'=0$, so $x=x', y=y'$, proving uniqueness. Conversely, if decomposition is unique, in particular $0$ can only decompose as $0+0$, implying no non-zero vector can be in both $X$ and $Y$ (so intersection is $\{0\}$); and certainly $X+Y$ must equal $V$ by the assumption that every vector can be expressed as such a sum.

The notion extends to more than two subspaces: We say $V = U_1 \oplus U_2 \oplus \cdots \oplus U_k$ if (1) $V = U_1 + U_2 + \cdots + U_k$ and (2) the intersection of any subcollection of the $U_i$'s is trivial (equivalently, the uniqueness of representing any vector as a sum of vectors from each $U_i$ holds). In practice, a simple criterion for a direct sum of multiple subspaces is that no nontrivial linear combination of vectors from different subspaces can yield zero except the trivial combination.

When $V$ is a direct sum of subspaces $X$ and $Y$, we sometimes call $X$ and $Y$ complementary subspaces. Each element of $V$ can be split into an $X$-part and a $Y$-part in a unique way. A classic example: in $\mathbb{R}^2$, let $X$ be the $x$-axis and $Y$ be the $y$-axis. Then indeed $\mathbb{R}^2 = X \oplus Y$ because any vector $(a,b)$ can be uniquely written as $(a,0)+(0,b)$ with $(a,0)\in X$ and $(0,b)\in Y$. The intersection $X \cap Y = \{\mathbf{0}\}$ (only the origin lies on both the $x$- and $y$-axes). If we had chosen $Y$ to be another line through the origin that is not the $y$-axis, say $Y$ is the line spanned by $(1,1)$, then $X \cap Y = \{0\}$ still (the $x$-axis and the line $y=x$ intersect only at $(0,0)$). Is $\mathbb{R}^2 = X \oplus Y$ in that case? Yes, because $(1,1)$ is not a multiple of $(1,0)$, so together those two directions span $\mathbb{R}^2$ and any $(a,b)$ has a unique decomposition $(a,b) = (c,0) + (d,d)$ for suitable $c,d$. If instead we chose $Y$ to be the line spanned by $(2,0)$, that would not give a direct sum decomposition with $X$ since $Y$ is actually the same subspace as $X$ (just a different basis vector for it), and $X \cap Y = X$ in that degenerate case (not trivial). There would be infinitely many ways to write a vector on the $x$-axis as $x + y$ with $x\in X, y\in Y$ because those are essentially the same subspace.

### Criterion for Direct Sum

**Criterion (for two subspaces):** $V = X \oplus Y$ if and only if $X+Y=V$ and $X \cap Y = \{0\}$. Often, it's easy to check $X+Y = V$ (span condition) and $X \cap Y = \{0\}$ (independence condition) to conclude a direct sum. In terms of dimensions, if $V = X \oplus Y$ in finite dimensions, then $\dim(V) = \dim(X) + \dim(Y)$ (since none of the dimension of $X$ is "wasted" in overlapping with $Y$). Conversely, if $\dim(X)+\dim(Y)=\dim(X+Y)$, it implies $X \cap Y$ must be trivial, hence $X+Y$ is direct.

### Examples

**Example 1:** Decomposing $\mathbb{R}^n$ into complementary subspaces. Consider $X=\{(x,0,0,\ldots,0)\}$ the $x_1$-axis in $\mathbb{R}^n$, and $Y=\{(0,x_2,x_3,\ldots,x_n)\}$ the subspace of vectors with first coordinate $0$. Then $X \cap Y = \{\mathbf{0}\}$ and $X+Y = \mathbb{R}^n$ (any vector splits into its first-coordinate part plus the rest). So $\mathbb{R}^n = X \oplus Y$. Here $Y = (\text{span of basis }e_2,\ldots,e_n)$ can be viewed as a complement of $X$ in $\mathbb{R}^n$ (and vice versa, $X$ is a complement of $Y$).

**Example 2:** Direct sum in matrix spaces. The space of all $n\times n$ real matrices, denoted $M_{n}(\mathbb{R})$, can be seen as the direct sum of two subspaces: $S =$ the space of symmetric matrices and $A =$ the space of skew-symmetric (anti-symmetric) matrices. Any matrix $M$ can be uniquely written as $M = S + A$ where $S = \frac{1}{2}(M + M^T)$ is symmetric and $A = \frac{1}{2}(M - M^T)$ is skew-symmetric. We have $M_{n}(\mathbb{R}) = S \oplus A$. Indeed, $S \cap A = \{\mathbf{0}\}$ (the only matrix that is both symmetric and skew-symmetric is the zero matrix), and clearly $S + A$ gives all matrices (by the formula above, every matrix is the sum of one symmetric and one skew matrix). This is a powerful decomposition in linear algebra and has practical uses (e.g. any square matrix's even and odd parts under transpose). The uniqueness of the decomposition is evident from the formula – it's essentially the projection onto symmetric vs skew components.

**Example 3:** Direct sum of more than two subspaces. Consider $\mathbb{R}^3$ and let $U_1 = \text{span}\{(1,0,0)\}$ (the $x$-axis), $U_2 = \text{span}\{(0,1,0)\}$ (the $y$-axis), and $U_3 = \text{span}\{(0,0,1)\}$ (the $z$-axis). Then $\mathbb{R}^3 = U_1 \oplus U_2 \oplus U_3$, since any vector $(a,b,c)$ decomposes uniquely as $(a,0,0)+(0,b,0)+(0,0,c)$. All pairwise intersections are trivial (each pair of axes only meet at the origin). This extends the idea of coordinate axes providing a direct sum decomposition of the space (which is essentially what a basis does). In general, if $\{\mathbf{v}_1,\ldots,\mathbf{v}_n\}$ is a basis of $V$, and we let $U_i = \text{span}\{\mathbf{v}_i\}$ (the one-dimensional subspace generated by $\mathbf{v}_i$), then $V = U_1 \oplus \cdots \oplus U_n$.

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

## Conclusion

The direct sum gives a rigorous way to split a vector space into independent parts. In ML terms, if certain feature sets or embedding components are independent (capturing orthogonal information), we can think of the overall representation as a direct sum of those feature subspaces. This viewpoint helps in understanding multi-component models. For example, one might say a model's embedding space factorizes into subspaces each encoding a different type of information (if true, that's beneficial because the information won't interfere). Indeed, researchers often describe multi-head attention as providing multiple representation subspaces for the model to attend to. By ensuring those subspaces are "independent" (linearly, to some extent), the model effectively uses a direct-sum-like structure to diversify what each head learns.

