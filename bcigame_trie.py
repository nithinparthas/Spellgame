# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 23:52:17 2020

@author: Nithin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:56:52 2020

@author: Nithin
"""

# NO IMPORTS!
import pickle
import numpy as np

def match(word, pattern):
    """Match a word to a pattern.
       pattern is a string, interpreted as explained below:
             * matches any sequence of zero or more characters,
             ? matches any single character,
             otherwise char in pattern char must equal char in word. 
    """
    # if the pattern is empty, only match empty word
    if len(pattern) == 0: return len(word) == 0
    # if the word is empty, only match "*"
    if len(word) == 0: return pattern == "*"
    
    # otherwise try to match first char in pattern
    if pattern[0] == "?" or pattern[0] == word[0]:
        # try to match pattern and word without first chars
        return match(word[1:], pattern[1:])
    elif pattern[0] == "*":
        # skip chars and try to match with rest of pattern
        for i in range(len(word)+1):
           if match(word[i:], pattern[1:]):
               return True
        return False
    else:
        return False
            
        
class Trie:
    ##################################################
    ## basic methods
    ##################################################

    def __init__(self):
        # is word if > 0
        self.frequency = 0
        # char : Trie
        self.children = {}

    def insert(self, word, frequency=0):
        """ add word with given frequency to the trie. """
        # end of the word, update freq for the node
        if len(word) == 1:
            # create children node if needed
            if word not in self.children:
                self.children[word] = Trie()
            self.children[word].frequency = frequency
    
        # descend through trie one char at a time
        else:
            # create children node if needed
            if word[0] not in self.children:
                self.children[word[0]] = Trie()
            self.children[word[0]].insert(word[1:], frequency)

    def find(self, prefix):
        """ return trie node for specified prefix, None if not in trie. """
        # return root if empty prefix
        if len(prefix) == 0:
            return 0
        
        # end of prefix, return the node
        elif len(prefix) == 1:
            if prefix in self.children:
                return self.children[prefix].frequency
            else:
                return 0
            # None if not in trie
            
        # descend through trie one character at a time
        else:
            if prefix[0] in self.children:
               return self.children[prefix[0]].find(prefix[1:])
            else:
                return 0
            # None if not in trie
            
    def findnode(self, prefix):
        """ return trie node for specified prefix, None if not in trie. """
        # return root if empty prefix
        if len(prefix) == 0:
            return 0
        
        # end of prefix, return the node
        elif len(prefix) == 1:
            if prefix in self.children:
                return self.children[prefix]
            else:
                return 0
            # None if not in trie
            
        # descend through trie one character at a time
        else:
            if prefix[0] in self.children:
               return self.children[prefix[0]].findnode(prefix[1:])
            else:
                return 0
            # None if not in trie

    def __contains__(self, word):
        """ is word in trie? return True or False. """
        trie = self.find(word)
        return trie is not None and trie.frequency >= 0

    def __iter__(self):
        """ generate list of (word,freq) pairs for all words in
            this trie and its children.  Must be a generator! """
        def helper(trie, prefix):
            # if word, then yield
            if trie.frequency > 0:
                    yield (prefix, trie.frequency)
            # visit all children
            for ch,child in trie.children.items():
                yield from helper(child, prefix + ch)
            
        return helper(self, "")
    

    ##################################################
    ## additional methods
    ##################################################

    def autocomplete(self, prefix, N):
        """ return the list of N most-frequently occurring words
            that start with prefix. """
        # find the node with prefix
        node = self.findnode(prefix)
        # return empty list if no such node
        if node is None: return []
        
        # generate list of all words with prefix
        # and sort in descending order

        words = list((freq, prefix+suffix) for suffix,freq in node)
        words = [word for word in words if word[1].endswith('_')]
        sumwords = sum([sublist[0] for sublist in words])
        
        words.sort(reverse=True)

        # return N most-frequent words
        return ([words[i] for i in range(min(N,len(words)))],sumwords)
        
    def autocorrect(self, prefix, N):
        """ return the list of N most-frequent words that start with
            prefix or that are valid words that differ from prefix
            by a small edit. """
       
        def add_word(string):
            """ add string to the set if in the trie and is word """
            if string in self:
                freq = self.find(string).frequency
                if freq > 0: words.add((freq, string))

        # find autocomplete words
        result = self.autocomplete(prefix, N)

        # if list is smaller than N, add eddited words
        C = len(result)
        if C < N:
            words = set()
            for i in range(len(prefix)):
                # single-char deletion
                delete = prefix[:i] + prefix[i+1:]
                add_word(delete)
                for ch in "abcdefghijklmnopqrstuvwxyz":
                    # single-char replacement
                    replace = prefix[:i] + ch + prefix[i+1:]
                    add_word(replace)
                    # single-char insertion
                    insert = prefix[:i] + ch + prefix[i:]
                    add_word(insert)
                if i < len(prefix)-1:
                    # char transposition
                    transpose = prefix[:i] + prefix[i+1] + prefix[i] + prefix[i+2:]
                    add_word(transpose)

            # sort in descending order and
            # add as many as needed to the list
            words = list(words)
            words.sort(reverse=True)
            for i in range(min(N-C,len(words))):
                result.append(words[i][1])
            
        return result

    def filter(self,pattern):
        """ return list of (word, freq) for all words in trie that match
            pattern.  pattern is a string, interpreted as explained below:
             * matches any sequence of zero or more characters,
             ? matches any single character,
             otherwise char in pattern char must equal char in word. """
        # iterate through all words and leave only which match
        return [(word, freq) for word,freq in self if match(word, pattern)]

def initialize_dist(bspace_char_flag, trie, charlist, declist, bspace_prob, totcharcnt, debug_wmodel, flag_word_complete):
    
    dist = [1.0/36.0]*36
    dist[34] = bspace_prob  # Probability of backspace
    sp_index = 0
    fs_index = 0
    declist_j = ''.join(declist[0:len(declist)]) # Create a joined version of declist

    # Get last index of '_' (end of word) in list
    index, num, denom = 0, 0, 0
    if len(declist_j) > 0 and '_' in declist_j: # Locate word space
       sp_index = len(declist_j) - declist_j[::-1].index('_') - 1  
    if len(declist_j) > 0 and '.' in declist_j:  # Locate period
       fs_index = len(declist_j) - declist_j[::-1].index('.') - 1  
    index = sp_index
    if fs_index > sp_index:  # Find space or period, whichever occurs last
        index = fs_index
        
    if index != 0:
        if index == len(declist_j)-1: # Case where '_' or '.' was just decoded
           cword = ''
           denom = totcharcnt
        else:
           cword = ''.join(declist_j[index+1:len(declist_j)])
           denom = trie.find(cword)  #Get total count
        
        if debug_wmodel:
          print("Cword is %s Index is %d declist is %s" %(cword,index, declist))
    else:
        if len(declist) == 0:  # Initial case right after startup when nothing has been decoded 
          cword = ''
          denom = totcharcnt
        else:   # For the case after startup, where we dont have a '_' yet
          cword = ''.join(declist[0:len(declist)])
    #      print("CWORD is %s" %cword)
          denom = trie.find(cword)
          
   # print("Index of _ is %d cword is %s declist=%s" %(index, cword, declist))      
    if debug_wmodel:
       print("Index of _ is %d cword is %s declist=%s" %(index, cword, declist))
       print(denom)
       
    if denom != 0:
      for i in range(len(charlist)-3):
         cword_c = str(cword)+charlist[i]
       
         num = trie.find(cword_c)
         if debug_wmodel:
            print("cword_c is %s, cword=%s" %(cword_c, cword))
            print(num, denom)
         if num == 0:
            dist[i] = 1e-6
         else:
            dist[i] = num/denom
      cword_c = str(cword) + '_'
      num = trie.find(cword_c)
      
      if debug_wmodel:
         print("cword_c is %s cword=%s num=%d" %(cword_c, cword, num))
         print(denom)
         
      dist[35] = num/denom
      if len(declist)> 0:
        if declist[-1] == '.':
                             # If a period was decoded, next is word space by assumption
           dist = [1e-6]*36       # So override all the computed probs for this special case
           dist[34] = bspace_prob  
           dist[35] = 1
        else:
           dist[33]=0.0482*dist[35]; # Following Dr. Speier's email about 4.8% of word spaces are sentence endings.
           dist[35]=0.9518*dist[35]; # Rest are word spaces
      
    words = []
    word_prob = [0.0]*5
    if flag_word_complete and not bspace_char_flag:
        if len(cword) > 0:
           words, tot_wd_freq = trie.autocomplete(cword,5)
           if len(words) >= 1:
             word_prob[0] = words[0][0]/tot_wd_freq # 0 is highest frequency word, goes to location dist[location of char '7']
             dist[charlist.index('7')] = 0
           if len(words) >= 2:
             word_prob[1]= words[1][0]/tot_wd_freq      # 1 is second-highest frequency word, goes to location dist[location of char '6']
             dist[charlist.index('6')] = 0
           if len(words) >= 3:
             word_prob[2]= words[2][0]/tot_wd_freq
             dist[charlist.index('5')] = 0
           if len(words) >= 4:
             word_prob[3]= words[3][0]/tot_wd_freq
             dist[charlist.index('4')] = 0
           if len(words) >= 5:
             word_prob[4]= words[4][0]/tot_wd_freq   
             dist[charlist.index('3')] = 0
           sum_word_prob = sum(word_prob)
           if sum_word_prob > 0:                         # This case when the word exists in the model
             word_prob = [(x*0.5)/sum_word_prob for x in word_prob]  
    
    sum_dist = sum(dist)
    if flag_word_complete and len(words) > 0 and sum_dist > 0 and not bspace_char_flag: # When word flag is on, then renormalize to 0.5 (split between character and word probabilities)
          dist = [(x*0.5)/sum_dist for x in dist] 
          if len(words) >= 1:
              dist[charlist.index('7')] = word_prob[0]
          if len(words) >= 2:
              dist[charlist.index('6')] = word_prob[1]
          if len(words) >= 3:
              dist[charlist.index('5')] = word_prob[2]
          if len(words) >= 4:
              dist[charlist.index('4')] = word_prob[3]
          if len(words) >= 5:
              dist[charlist.index('3')] = word_prob[4]           
    elif sum_dist > 0:                          
          dist = [x/sum_dist for x in dist]     # Renormalize to a total prob of 1         
    w = []
    for w1 in words:
        w.append(w1[1]) 
    loc=np.argsort(dist)[::-1]
    loc = loc.tolist()  # convert nparray to list
      
    return(dist, loc, w, cword)

def get_rows_cols_diag(loc, charlist, flashboard_type):
    charlist_diag, order_diag = [], []
    if (flashboard_type == 1):
      charlist_diag = [charlist[loc[0]], charlist[loc[6]], charlist[loc[16]], charlist[loc[24]], charlist[loc[30]], charlist[loc[34]], 
                       charlist[loc[11]], charlist[loc[1]], charlist[loc[7]], charlist[loc[17]], charlist[loc[25]], charlist[loc[31]],
                       charlist[loc[20]], charlist[loc[12]], charlist[loc[2]], charlist[loc[8]], charlist[loc[18]], charlist[loc[26]], 
                       charlist[loc[27]], charlist[loc[21]], charlist[loc[13]], charlist[loc[3]], charlist[loc[9]], charlist[loc[19]],
                       charlist[loc[32]], charlist[loc[28]], charlist[loc[22]], charlist[loc[14]], charlist[loc[4]], charlist[loc[10]], 
                       charlist[loc[35]], charlist[loc[33]], charlist[loc[29]], charlist[loc[23]], charlist[loc[15]], charlist[loc[5]]]
      order_diag = [loc[0],loc[6],loc[16],loc[24],loc[30],loc[34],loc[11],loc[1],loc[7],loc[17],loc[25],loc[31],
                   loc[20],loc[12],loc[2],loc[8],loc[18],loc[26],loc[27],loc[21],loc[13],loc[3],loc[9],loc[19],
                   loc[32],loc[28],loc[22],loc[14],loc[4],loc[10],loc[35],loc[33],loc[29],loc[23],loc[15],loc[5]]
 
    return(charlist_diag, order_diag)      

def get_total_charcnt(charlist, trie):
    
    i = 0
    totcharcnt = 0
    while i < len(charlist):
        totcharcnt = totcharcnt + trie.find(charlist[i])
        i = i + 1
    
    return(totcharcnt)
      
    
 

