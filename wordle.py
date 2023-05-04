import numpy as np
import sys
from collections import Counter
import ipdb
import warnings
warnings.filterwarnings("error")

"""
Sting Fight!
'TLDR: pandas are Jedi; numpy are the hutts; and python is the galactic empire.'
https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
"""

# "wordlist-wordle-answers.txt"
# "wordlist-both-sets-combined.txt"
# "wordlist-allowed-guesses.txt"
words = np.loadtxt("wordlist-allowed-guesses.txt",dtype=np.bytes_)

alphabet = b"abcdefghijklmnopqrstuvwxyz"
allchars = b""
for w in words: allchars += w
ct = Counter(allchars)

# Example guess and mark histories
guess_hist = b"toots boogi grama ambid aliph aleth"
mark_hist  = b"KKKKK KKKKK KKYKG GKKKK GGKYY GGKKY"



# Filter all possible words given history of guesses and marks
def shrink(guess_hist, mark_hist):
  guess_hist = np.frombuffer(guess_hist, dtype='S1')
  mark_hist = np.frombuffer(mark_hist, dtype='S1')

  black  = set(guess_hist[mark_hist==b'K'])
  yellow = set(guess_hist[mark_hist==b'Y'])
  green  = set(guess_hist[mark_hist==b'G'])

  ## filter black
  possible_solutions = [w for w in words if not any([l in w for l in black])] 
  # ## filter green and yellow
  possible_solutions = [w for w in possible_solutions if all([l in w for l in green | yellow])]
  # ## filter green and yellow (positions)

  guess_hist = guess_hist[guess_hist!=b' '].reshape([-1,5])
  mark_hist  = mark_hist[mark_hist!=b' '].reshape([-1,5])

  def filt(w):
    w = np.frombuffer(w,dtype='S1')
    for i,l in enumerate(w):
      for j in range(mark_hist.shape[0]):
        # ipdb.set_trace()
        if mark_hist[j,i]==b'G' and guess_hist[j,i]!=l: return False
        if mark_hist[j,i]==b'Y' and guess_hist[j,i]==l: return False
    return True
        
  possible_solutions = [w for w in possible_solutions if filt(w)]
  return possible_solutions

# Mark a single guess
def markGuess(guess, soln):
  # guess = bytes(guess,"utf8")
  mark  = b""
  for i,g in enumerate(guess):
    if g==soln[i]: 
      mark += b'G'
    elif g in soln: 
      mark += b'Y'
    else:
      mark += b'K'
  return mark

# Compute best guess given guess+mark history using brute force on small
# selection of guess words.
def bestGuess_randomSample(guess_hist, mark_hist):
  ps0 = shrink(guess_hist,mark_hist)
  
  print(f"{len(ps0)} valid words remain")
  if len(ps0)<20: print(ps0)

  guess_costs = []
  # for g in ps0:
  for g in np.random.choice(words,40):
    ## try 40 different possible guesses
    soln_costs = []
    for s in ps0:
      ## for each potential solution, how much would this guess shrink the pool?
      newmark = markGuess(g,s)
      gh  = guess_hist + g
      mh  = mark_hist + newmark
      ps1 = shrink(gh,mh)
      soln_costs.append(len(ps1))

    # The cost of a guess is the size of potential answer set after shrinking
    # averaged over all possible answers.
    guess_costs.append(np.mean(soln_costs))
    print(g.decode(), guess_costs[-1])

# Compute best guess given guess+mark history using greedy entropy-maximizing
# letter tree.
# 
# Fast enough to give results on full word list: optimal splits
# are "e,r,a,s,t" which gives us "tears", "stare" , "rates" , and "aster" as
# solutions.
def bestGuess_letterTree(guess_hist, mark_hist):
  """

  Suggest a word by greedy maximization on a letter tree. We split greedily on
  letters that maximize information gain[^1], while simultaneously filtering
  out words that don't contain the letter. NOTE: It is unlikely that greedily
  selecting the first five letters which maximize infogain will product a
  valid word!

  [^1]: The Nth layer of the binary tree corresponds to the Nth letter in the
  word. Each node partitions remaining words into sets with/without the
  letter. A good letter choice will maximize the entropy of the distribution
  across all nodes in a layer.



                                               ┌ ─ ─ ─ ─                       
                                                 capit  │                      
                                               │ roman                         
                                                 apple  │                      
                                               │ lifer                         
                                                 gimbl  │                      
                                               │ leggy                         
                                                 other  │                      
                                               └ ─ ─ ─ ─                       
                          ┌ ─ ─ ─ ─                 │                          
                            roman  │                │                          
                          │ apple                   ▼                ┌ ─ ─ ─ ─ 
                            lifer  │              ┌───┐                       │
                          │ gimbl   ────┐         │ c │         ┌────│ capit   
                            leggy  │    │         └───┘         │             │
                          │ other       │                       │    └ ─ ─ ─ ─ 
                           ─ ─ ─ ─ ┘    │    ┌───┐     ┌───┐    │              
                                        └───▶│ r │     │ r │◀───┘              
                                             └───┘     └───┘                   
              ┌ ─ ─ ─ ─                                                        
                apple  │           ┌───┐     ┌───┐     ┌───┐     ┌───┐         
              │ gimbl   ──────────▶│ a │┌───▶│ a │     │ a │     │ a │         
                leggy  │           └───┘│    └───┘     └───┘     └───┘         
              └ ─ ─ ─ ─                 │                                      
                            ┌───┐┌───┐  │┌───┐┌───┐   ┌───┐┌───┐    ┌───┐┌───┐ 
                            │ n ││ n │  ││ n ││ n │   │ n ││ n │    │ n ││ n │ 
                            └───┘└───┘  │└───┘└───┘   └───┘└───┘    └───┘└───┘ 
                                        │                                      
                                   ┌ ─ ─ ─ ─                                   
                                     roman  │                                  
                                   │ lifer                                     
                                     other  │                                  
                                   └ ─ ─ ─ ─                                   


  """
  valid = shrink(guess_hist, mark_hist)

  print(f"Size of valid is {len(valid)}")
  if len(valid)<20: print(valid)

  ## trim alphabet down to just the critical letters which vary between words
  unio  = set.union(*[set(w) for w in valid])
  isect = set.intersection(*[set(w) for w in valid])
  # _alphabet = set(b"abcdefghijklmnopqrstuvwxyz")
  _alphabet = ''.join([chr(x) for x in list(unio - isect)])

  print(f"alphabet is: {_alphabet}")

  if len(_alphabet) <= 5:
    print("Alphabet too small... ")
    print("valid: ")
    print(valid)
    return
    # return valid

  valid = np.array(valid, dtype='S5')
  guess = np.array(words)

  ## build the tree
  # alphabet = _alphabet.copy()
  alphabet = list(unio - isect)
  letters = []

  layer = -1; dolayers = True;
  while dolayers:
    layer += 1

    entropies = []
    for l in alphabet: ## brute force check every letter

      splits = []
      for j in range(2**layer): ## 2**i nodes at layer i. compute fraction of words in node at layer i+1 from layer i.

        ## filter valid words based on location in tree
        _valid = valid.copy()
        branches = f"{j:05b}"
        for k,c in enumerate(letters):
          # print(chr(c),end='')
          # ipdb.set_trace()
          if branches[-k]=='0':
            _valid = _valid[np.char.find(_valid, bytes(chr(c),'utf8')) != -1]
          else:
            _valid = _valid[np.char.find(_valid, bytes(chr(c),'utf8')) == -1]
          if len(_valid)==0: break

        ## count remaining valid words
        s0 = len(_valid)
        s1 = len([w for w in _valid if l in w])
        splits.append(s1)
        splits.append(s0 - s1)

      ## print character stats      
      # print(chr(l))
      splits = np.array(splits)
      # print(splits)
      splits = splits / splits.sum()
      m = splits>0
      _splits = splits.copy()
      _splits[m] = np.log(_splits[m])*_splits[m]
      entro = -np.sum(splits)
      # entro  = -np.sum(np.where(splits>0 , np.log(splits)*splits , 0))
      # print(f"Entropy is {entro}")
      entropies.append(entro)

    ## pick best split character
    idxs = np.argsort(entropies)
    for i in range(1,10):
      bestchar = alphabet[idxs[-i]]
      lenguess = len(guess)
      _guess = guess[np.char.find(guess, bytes(chr(bestchar),'utf8')) != -1]
      print(f"Using letter `{chr(bestchar)}` guesses trimmed from {lenguess} to {len(_guess)}.")

      if len(_guess)==0:
        continue
      elif len(_guess)<10:
        letters.append(bestchar)
        guess = _guess
        dolayers = False ## break outer loop
        break
      else:
        letters.append(bestchar)
        guess = _guess

  print(f"Our letters are: {''.join([chr(x) for x in letters])}")
  print(f"And our guess words are: {guess}")
  return guess

# Quadratic solution. This is a strict improvement over the N^3 solution `bestGuess_randomSample`,
# but it's still brute force in that it tries every possible word.
def bestGuess_MarkEntropy(guess_hist, mark_hist):
  """
  ==WORST WORD==
  1.483 'fuzzy'

  ==BEST WORDS==
  4.212 'tears' 
  4.203 'rates'
  4.195 'aries'
  4.168 'tales'
  4.161 'nares'
  4.161 'raise'
  4.143 'slate'
  4.139 'stare'
  4.123 'snare'
  """

  valid = shrink(guess_hist, mark_hist)

  if len(valid)<10: 
    print(valid)
    return


  # TODO: Ensure that we always have each of 3^5 possible marks in the
  # distribution. EDIT: This doesn't matter because the empty marks have zero
  # probability and thus DO NOT CHANGE ENTROPY
  def entropy(lst):
    lst = np.array(lst)
    lst = lst / lst.sum()
    m = lst>0
    lst[m] = lst[m]*np.log(lst[m])
    return -lst.sum()

  # for every possible guess, and every possible solution, mark the guess
  # against that solution. Bin and count the marks (3^5 possible marks). The
  # heuristic is that the best guess is the one which _maximizes_ the entropy
  # of this mark distribution. This avoids an extra pass over the word list like
  # bestGuess_randomSample().
  entropies = []
  countlist = []
  for i,g in enumerate(words):
    if i%100==0: print(f"i = {i}", end='\r')
    counts = Counter([markGuess(g,s) for s in valid])
    countlist.append(counts)
    entropies.append(entropy(list(counts.values())))

  idxs = np.argsort(entropies)

  print("Guess , Mark Entropy")
  for i in range(1,10):
    e = entropies[idxs[-i]]
    w = words[idxs[-i]]
    print(f"{w.decode('utf8')} , {e:.3f}")


# Entry point
def play_wordle(soln=None):

  intro = """
  Let's Play Wordle!
  This version encourages cheating. We will recommend very good guesses to you,
  and you can always look at the list of remaining possible answers with `P`.
  """
  intro = '\n'.join([x.lstrip() for x in intro.split('\n')])
  print(intro)

  if type(soln) is str: soln = soln.encode('utf-8')

  if soln not in words:
    print(soln, "not in words")
    soln = np.random.choice(words)

  guess_hist = b""
  mark_hist = b""

  n_guess = 0
  while n_guess < 6:
    guess = input("Guess a word: ")
    guess = bytes(guess,"utf8")

    if guess==b'P': 
      print(shrink(guess_hist,mark_hist))
      continue

    n_guess += 1
    mark = markGuess(guess,soln)
    guess_hist += b" " + guess
    mark_hist += b" " + mark
    print()
    print(guess_hist.decode("utf8"))
    print(mark_hist.decode("utf8"))
    print()
    
    if guess==soln:
      print(f"Correct, the word is : {soln.decode()} !")
      print(f"You Won in {n_guess} guesses.")
      break

    print("-- Max Mark Entropy Clue --")
    bestGuess_MarkEntropy(guess_hist, mark_hist)
    # print("-- Greedy Letter Clue --")
    # bestGuess_letterTree(guess_hist, mark_hist)
    # print("-- Best among random sample --")
    # bestGuess_randomSample(guess_hist,mark_hist)

  if n_guess >= 6:
    print(f"You lose :( The correct answer was {soln.decode()} .")


if __name__=='__main__': 
  if len(sys.argv)>=2: 
    play_wordle(sys.argv[1])
  else:
    play_wordle()


