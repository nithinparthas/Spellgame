# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 09:13:39 2020

@author: Nithin
"""

import sys

from crossword import *
import itertools
import copy


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # Loop through each variable and initialise empty set for words that violate node consistency
        for variable in self.domains:
            words_to_remove = set()
            
            # Loop through each word in variable's domain and check for node consistency
            for word in self.domains[variable]:
                if len(word) != variable.length:
                    words_to_remove.add(word)
            
            # For all words that violate node consistency, remove from variable's domain
            for word in words_to_remove:
                self.domains[variable].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.
        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]

        # Need to check x's domain if there is overlap between x and y
        if overlap is not None:

            # Initialise set of words to remove from x's domain
            words_to_remove = set()

            # Check each word in x's domain
            for x_word in self.domains[x]:
                overlap_char = x_word[overlap[0]]
                corresponding_y_chars = {w[overlap[1]] for w in self.domains[y]}
                
                # If there's no possible value in y's domain that doesn't cause conflict,
                # need to remove that word from x's domain
                if overlap_char not in corresponding_y_chars:
                    words_to_remove.add(x_word)
                    revised = True

            # Remove words from x's domains
            for word in words_to_remove:
                self.domains[x].remove(word)

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.
        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # Start with all arcs (tuple of varibales with overlap) if none provided
        if arcs is None:
            queue = list(itertools.product(self.crossword.variables, self.crossword.variables))
            queue = [arc for arc in queue if arc[0] != arc[1] and self.crossword.overlaps[arc[0], arc[1]] is not None]
        else:
            queue = arcs

        # Repeat while queue is non-empty
        while queue:
            arc = queue.pop(0)
            x, y = arc[0], arc[1]

            # Make variable x arc consistent with variable y
            if self.revise(x, y):
                
                # If domain is empty, problem is unsolvable
                if not self.domains[x]:
                    print("ending ac3")
                    return False
                
                # Append arc to queue after making change to domain (to ensure other arcs stay consistent)
                for z in (self.crossword.neighbors(x) - {y}):
                    queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # Assignment complete if each variable has a key in the assignment dictionary
        # and its corresponding value is a non-empty string
        if set(assignment.keys()) == self.crossword.variables and all(assignment.values()):
            return True
        else:
            return False
        
    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Inconsistent if not all values are distinct
        if len(set(assignment.values())) != len(set(assignment.keys())):
            return False

        # Inconsistent if not every value is correct length
        if any(variable.length != len(word) for variable, word in assignment.items()):
            return False

        # Inconsistent if there are conflicts between neighbouring variables
        for variable, word in assignment.items():
            for neighbor in self.crossword.neighbors(variable).intersection(assignment.keys()):
                overlap = self.crossword.overlaps[variable, neighbor]
                if word[overlap[0]] != assignment[neighbor][overlap[1]]:
                    return False

        # Otherwise consistent
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # Initialise dictionary to store the number of choices eliminated for each word in var's domain
        num_choices_eliminated = {word: 0 for word in self.domains[var]}

        # Set of neighbors of var
        neighbors = self.crossword.neighbors(var)

        # Loop through each word in var's domain
        for word_var in self.domains[var]:
            
            # Loop through each of var's unassigned neighbors
            for neighbor in (neighbors - assignment.keys()):
                overlap = self.crossword.overlaps[var, neighbor]

                # Loop through each word in neighbor's domain
                for word_n in self.domains[neighbor]:
                    
                    # Increment number of choices eliminated if there is conflict between
                    # word in var's domain and word in neighbor's domain
                    if word_var[overlap[0]] != word_n[overlap[1]]:
                        num_choices_eliminated[word_var] += 1

        # Sort var's domain in ascending order of the number of values they rule out for neighboring variables
        sorted_list = sorted(num_choices_eliminated.items(), key=lambda x:x[1])
        return  [x[0] for x in sorted_list]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_variables = self.crossword.variables - assignment.keys()

        # Number of remaining values in each variable's domain
        num_remaining_values = {variable: len(self.domains[variable]) for variable in unassigned_variables}
        sorted_num_remaining_values = sorted(num_remaining_values.items(), key=lambda x: x[1])

        # If there is no tie, return variable with minimum number of remaining values in domain
        if len(sorted_num_remaining_values) == 1 or sorted_num_remaining_values[0][1] != sorted_num_remaining_values[1][1]:
            return sorted_num_remaining_values[0][0]
        
        # If there is a tie, return variable with highest degree
        else:
            num_degrees = {variable: len(self.crossword.neighbors(variable)) for variable in unassigned_variables}
            sorted_num_degrees = sorted(num_degrees.items(), key=lambda x: x[1], reverse=True)
            return sorted_num_degrees[0][0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.
        `assignment` is a mapping from variables (keys) to words (values).
        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            test_assignment = copy.deepcopy(assignment)
            test_assignment[var] = value
            if self.consistent(test_assignment):
                assignment[var] = value
                result = self.backtrack(assignment)
                if result is not None:
                    return result
            assignment.pop(var, None)
        return None

def generatepzl():

    # Check usage
#    if len(sys.argv) not in [3, 4]:
#        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
#    structure = sys.argv[1]
#    words = sys.argv[2]
#    output = sys.argv[3] if len(sys.argv) == 4 else None
    structure = 'structure/structure.txt'
    words = 'words/words.txt'
    output = 'output.png'

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)
        pzl_words = []
        for variable, word in assignment.items():
           pzl_words.append(word.lower())
        
        disp_words = [
            [None for _ in range(crossword.width)]
            for _ in range(crossword.height)
        ]
        map_words = [
            [0 for _ in range(crossword.width)]
            for _ in range(crossword.height)
        ] 
        orig_words = []
        for variable, word in assignment.items():
            direction = variable.direction
            orig_word = []
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
#                if k == 0 or k == len(word)-1:
                if k == 0:
                    disp_words[i][j] = word[k] 
                    map_words[i][j] = 0
                else:
                    disp_words[i][j] = '_'
                    map_words[i][j] = 1
                orig_word.append(disp_words[i][j])
            orig_words.append(''.join(orig_word))
    spelled_chars = sum(sum(x) for x in map_words)            
    return(creator.letter_grid(assignment), assignment, crossword.height, crossword.width, pzl_words, disp_words, map_words, spelled_chars, orig_words)

def update_tag_char(grid, pzl_word, map_words, display_words, dec, cindex):
    
   for variable, word in grid.items():
         direction = variable.direction
         dec = ''.join(dec)
         
         if word.lower() == pzl_word:
               for k in range(len(pzl_word)):
                  i = variable.i + (k if direction == Variable.DOWN else 0)
                  j = variable.j + (k if direction == Variable.ACROSS else 0)
#                  print(dec, k, word, i, j, map_words[i][j], pzl_word, pzl_word[k])
                  if map_words[i][j] == 1:
                     if k < len(dec):
                        if dec[k] == pzl_word[k]:
                          display_words[i][j] = dec[k].upper() 
                          map_words[i][j] = 0

   return(map_words, display_words) 

def get_word_details(declist, orig_word):
    word_details = orig_word
    word_details = ['*' if x == '_' else x for x in word_details]
    for i in range(min(len(orig_word),len(declist))):
        word_details[i] = declist[i].upper()
    return(word_details)         
    
def find_and_tag_char(grid, word_decoded, map_words, display_words, windex, cindex, full_pzl, pzl_words, declist):
   

  if windex < len(pzl_words):
      cindex_found, simdone = 0, 0
      pzl_word = pzl_words[windex]
  else:
      cindex_found, simdone = 1, 1
      pzl_word = pzl_words[len(pzl_words)-1]
  
  if word_decoded and windex < len(pzl_words):
      cindex = 0
      declist = []
  cindex = len(declist)
#  print("WINDEX, CINDEX IS %d %d" %(windex, cindex))
  windex_change = 0
  while not cindex_found:
    for variable, word in grid.items():
         direction = variable.direction              
         if word.lower() == pzl_word and not cindex_found:
#            for k in range(len(word)):
            k = cindex
            windex_change = 0
            while k < len(word) and not cindex_found and not windex_change:
               k = cindex
               i = variable.i + (k if direction == Variable.DOWN else 0)
               j = variable.j + (k if direction == Variable.ACROSS else 0)
 #              print(word, i, j, k, pzl_word, windex, cindex, cindex_found, map_words[i][j])
               
               if map_words[i][j] == 1 and not cindex_found:
                 display_words[i][j] = '*'
                 cchar = word[cindex]  
 #                map_words[i][j] = 0
                 cindex_found = 1
               elif not cindex_found:
                 if cindex < len(word)-1:
                   declist.append(word[cindex].lower())
                   cindex = cindex + 1
                 elif windex < len(pzl_words)-1:
                   windex = windex + 1
                   cindex = 0
                   pzl_word = pzl_words[windex]
                   declist = []
                   windex_change = 1
                 else:
                   print("GAME DONE")
                   simdone = 1
                   cindex_found = 1
            k = k + 1

  return(map_words, display_words, windex, cindex, pzl_word, declist, simdone)                   
     