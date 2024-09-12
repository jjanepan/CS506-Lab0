## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    ### YOUR CODE HERE
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = cosine_similarity(vector1, vector2)
    
    expected_result = 0.0
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    target_vector = np.array([1, 2, 3])  # Use parentheses instead of square brackets
    vectors = [
        np.array([4, 5, 6]),  # Add commas between elements
        np.array([1, 0, 0]),
        np.array([0, 1, 0])
    ]
    
    result = nearest_neighbor(target_vector, vectors)
    
    expected_index = 0  # The index of the vector [4, 5, 6], which is closest to [1, 2, 3] in terms of cosine similarity
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
