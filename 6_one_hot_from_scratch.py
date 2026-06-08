import numpy as np

def one_hot_encode(data):
    #as first occurance
    seen = set()
    categories = []
    for val in data:
        if val not in seen:
            seen.add(val)
            categories.append(val)
    # print(len(categories))

    #build category to index map 
    cat_to_idx = {cat:idx for idx,cat in enumerate(categories)}
    # print(cat_to_idx)

    encoded = []
    for val in data:
        row = [0.0] *len(categories) # empty rows of zeroes of unique categories
        # print(row)
        row[cat_to_idx[val]] = 1.0
        encoded.append(row)

    return categories,encoded


#given
if __name__ == "__main__":
    column = ["cat","dog","cat","bird","dog","cat"]
    matrix,labels = one_hot_encode(column)
    print(labels)
    print(matrix)