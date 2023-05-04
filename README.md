A simple implementation of the Wordle game to test guess suggestion algorithms.

```
~/wordle> python wordle.py

Let's Play Wordle!
This version encourages cheating. We will recommend very good guesses to you,
and you can always look at the list of remaining possible answers with `P`.

Guess a word: tares

 tares
 KKKGY

-- Max Mark Entropy Clue --
Guess , Mark Entropy
soldi , 3.008
lound , 2.962
indol , 2.944
sloid , 2.937
nould , 2.931
sloyd , 2.903
doily , 2.900
unsod , 2.897
unlid , 2.886

Guess a word: soldi

 tares soldi
 KKKGY YGKKK

-- Max Mark Entropy Clue --
[b'cosec', b'cosey', b'hosen', b'hosey', b'mosey', b'posey']
Guess a word: hosey

 tares soldi hosey
 KKKGY YGKKK KGGGG

-- Max Mark Entropy Clue --
[b'cosey', b'mosey', b'posey']
Guess a word: mosey

 tares soldi hosey mosey
 KKKGY YGKKK KGGGG GGGGG

Correct, the word is : mosey !
You Won in 4 guesses.
```