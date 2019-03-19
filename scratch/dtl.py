#pseudocode
"""
function dtl(examples, attributes, parent_examples) returns tree
    if len(examples) == 0:
        return plurality_value(parent_examples)
    elif same_classification(examples):
        return classification(examples)
    elif len(attributes) == 0:
        return plurality_value(parent_examples)
    else:
        A = max_info_attr(examples)
        tree = Tree(A) #new tree with test A
        values = attr_values(A)
        for each value in values:
            exs = get_attr_val_examples(examples, A, value)

            subtree = dtl(exs, attributes, examples)
"""