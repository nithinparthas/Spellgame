# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:06:46 2020

@author: Nithin
"""
import math

# Example Huffman coding implementation
# Distributions are represented as dictionaries of { 'symbol': probability }
# Codes are dictionaries too: { 'symbol': 'codeword' }

def huffman(p):
    '''Return a Huffman code for an ensemble with distribution p.'''
    sum_val = sum(p.values())
    if not math.isclose(sum_val,1.0):
       print("ERROR: PROBABILITY SUM is %f" %(sum_val))
#    assert(sum(p.values()) == 1.0) # Ensure probabilities sum to 1

    # Base case of only two symbols, assign 0 or 1 arbitrarily
    if(len(p) == 2):
        return dict(zip(p.keys(), ['0', '1']))

    # Create a new distribution by merging lowest prob. pair
    p_prime = p.copy()
    a1, a2 = lowest_prob_pair(p)
    p1, p2 = p_prime.pop(a1), p_prime.pop(a2)
    p_prime[a1 + a2] = p1 + p2

    # Recurse and construct code on new distribution
    c = huffman(p_prime)
    ca1a2 = c.pop(a1 + a2)
    c[a1], c[a2] = ca1a2 + '0', ca1a2 + '1'

    return c

def lowest_prob_pair(p):
    '''Return pair of symbols from distribution p with lowest probabilities.'''
    assert(len(p) >= 2) # Ensure there are at least 2 symbols in the dist.

#    sorted_p = sorted(p.items(), key=lambda (i,pi): pi)
    sorted_p = sorted(p.items(), key=lambda x: x[1])
    return sorted_p[0][0], sorted_p[1][0]