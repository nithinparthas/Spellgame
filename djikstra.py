# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:17:24 2020

@author: Nithin
"""
def djikstra(trie, trigram_trie, lword, charset, flag_store_djikstra):

 # chk_word = trie.autocomplete(word)
  depth = 5
   
  len_charset = len(charset)-3 # Don't need to find prob for backspace, period and wordspace is done seperately
  word_list, word_pr = '', []

  if len(lword) > 0:

    oldstoreword, newstoreword, oldweight, newweight = ['']*len_charset, ['']*len_charset, [1]*len_charset, [1]*len_charset
    maxword = ''
  
    for i in range(depth): #depth is the max search depth
      for j in range(len_charset): # Go through all the characters in the set (nodes in Djikstra)
          maxweight = 0
          pr_c = smooth_kn(oldstoreword[k], trie, trigram_trie, charset, pr_c,totcharcnt, tottrigramcharcnt)
          for k in range(len_charset): # For a given node, search through all the characters for min weight
                          
             metric = oldweight[k]*pr_c[k]
             chk_word = oldstoreword[k]+charset[j]
             if metric > maxweight:
                maxweight = metric
                maxword = chk_word

             if chk_word not in word_list:
                word_list.append(chk_word)
                word_pr.append(metric*pr_c)
          newweight[j] = maxweight
          newstoreword[j] = maxword;        

      oldweight = newweight
      oldstoreword = newstoreword

    word_list, word_pr = zip(*[(x, y) for x, y in sorted(zip(word_pr, word_list))]) # Sor the two lists togother
    if len(word_list) > 5:
          word_list = word_list[-5:-1]
          word_pr = word_pr[-5:-1]
          if word_pr < 0:
              print("WARNING, negative probability")
    if len(word_list) > 0:
         word_list= word_list + '_'
     
    return(word_list, word_pr)