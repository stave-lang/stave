## Task

Implement a binary search tree of `u32` keys stored in a `std.vec` vector of nodes, where each node
holds its key and the indexes of its children as `Option<usize>`. Insert 50, 30, 70, 20, 40, 60,
80, 35, 45, 65 in this order. Print the in-order traversal (recursive), the height (the number of
nodes on the longest root-to-leaf path), and whether 65 and 66 are in the tree.

## Output

```text
inorder: 20 30 35 40 45 50 60 65 70 80
height 4
has 65: true
has 66: false
```
