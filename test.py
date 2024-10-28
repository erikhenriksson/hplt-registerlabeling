labels_structure = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}

# Generate the list of all labels
labels_all = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]

# Create a mapping from child labels to parent labels
child_to_parent = {}
for parent, children in labels_structure.items():
    for child in children:
        child_to_parent[child] = parent

# Create a mapping from labels to their indices
label_to_index = {label: idx for idx, label in enumerate(labels_all)}

# Suppose labels_indexes is given as follows (example)
labels_indexes = [0] * len(labels_all)
# Let's say 'it' and 'ne' are present
labels_indexes[label_to_index["it"]] = 1
labels_indexes[label_to_index["sr"]] = 1

# Ensure parent labels are present when child labels are present
for i, label in enumerate(labels_all):
    if labels_indexes[i] == 1 and label in child_to_parent:
        parent_label = child_to_parent[label]
        parent_index = label_to_index[parent_label]
        labels_indexes[parent_index] = 1

# labels_indexes now correctly reflects parent labels being present
print(labels_indexes)
