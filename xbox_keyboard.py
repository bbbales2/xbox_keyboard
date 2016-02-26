#%%
# I stole the pygame/TextPrint/setup code straight from http://www.pygame.org/docs/ref/joystick.html
#

import pygame
import numpy
import bisect
import scipy.interpolate
import scipy.cluster
import collections
import nltk
import re
import os
import sys
# import InputDevice, ecodes

# This should be the path to the current directory
base = '/home/bbales2/xbox_keyboard/'#os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(os.path.join(base, '345.txt')):
    import urllib2
    print "Corpora doesn't exist... Downloading {0}".format('http://www.gutenberg.org/ebooks/345.txt.utf-8')
    url = urllib2.urlopen('http://www.gutenberg.org/ebooks/345.txt.utf-8')
    f = open(os.path.join(base, '345.txt'), 'w')
    f.write(url.read())
    f.close()
    print "Download finished"

print "Building dictionary"
sys.stdout.flush()

#Import our dictionary
f = open(os.path.join(base, '345.txt'))
data = unicode(f.read(), 'utf-8')
#print data
tokens = nltk.word_tokenize(data)
f.close()

words = []
clear = re.compile('[^A-Za-z]')
number = re.compile('[0-9]')

# We only want letters
# Get rid of words with any numbers
# Delete any other errant characters
for token in tokens:
    if number.match(token):
        continue

    filtered = clear.sub('', token).lower()
    words.append(filtered)

# We only want the most frequent words
# wbL is a dictionary mapping word lengths to a list of words of that length
# lprobs is a dictionary mapping word names to empirical probabilities
counts = collections.Counter()
pcounts = collections.Counter()
lcounts = collections.Counter()

for word in words:
    counts[word] += 1

pruned = []
for word in words:
    if counts[word] > 1:
        pruned.append(word)
        pcounts[word] = counts[word]

wbL = {}

for word in pruned:
    if len(word) not in wbL:
        wbL[len(word)] = set()

    wbL[len(word)].add(word)

    for w in word:
        lcounts[w] += 1.0

# wprobs is word probabilty given word length
wprobs = {}

for length in wbL:
    wprobs[length] = {}

    totalWs = 0.0
    for w in wbL[length]:
        totalWs += pcounts[w]

    for w in wbL[length]:
        wprobs[length][w] = pcounts[w] / totalWs

# unconditional letter probabilities
lprobs = {}

totals = sum(lcounts.values())
for l in lcounts:
    lprobs[l] = lcounts[l] / totals

print "Setting up keyboard"

lLetters = ['qwert',
            'asdfg',
            'zxcvb']

rLetters = ['yuiop',
            'hjkl',
            'nm']

dx = 1.0 / 6.0

sigma = 0.2

lLetterMeans = []
rLetterMeans = []

for i, row in enumerate(lLetters):
    uy = (i + 1) * 0.25
    for j, letter in enumerate(row):
        ux = i * dx * 0.1 + (j + 1) * dx

        lLetterMeans.append((letter, ux, uy))

for i, row in enumerate(rLetters):
    uy = (i + 1) * 0.25
    for j, letter in enumerate(row):
        ux = i * dx * 0.1 + (j + 1) * dx

        rLetterMeans.append((letter, ux, uy))
        
lMeans = numpy.mean([(ux, uy) for letter, ux, uy in lLetterMeans], axis = 0)
rMeans = numpy.mean([(ux, uy) for letter, ux, uy in rLetterMeans], axis = 0)

lMaxR = numpy.max(numpy.linalg.norm([(ux - lMeans[0], uy - lMeans[1]) for letter, ux, uy in lLetterMeans], axis = 1), axis = 0)
rMaxR = numpy.max(numpy.linalg.norm([(ux - rMeans[0], uy - rMeans[1]) for letter, ux, uy in rLetterMeans], axis = 1), axis = 0)

lLetterMeans = [(letter, (ux - lMeans[0]) / lMaxR, (uy - lMeans[1]) / lMaxR) for letter, ux, uy in lLetterMeans]
rLetterMeans = [(letter, (ux - rMeans[0]) / rMaxR, (uy - rMeans[1]) / rMaxR) for letter, ux, uy in rLetterMeans]

dists1 = { 0 : {}, # 0 is for lpresses, 1 is for rpresses!
           1 : {} }

for letter, ux, uy in lLetterMeans:
    dists1[0][letter] = scipy.stats.multivariate_normal([ux, uy], [[sigma**2, 0.0], [0.0, sigma**2]])

for letter, ux, uy in rLetterMeans:
    dists1[1][letter] = scipy.stats.multivariate_normal([ux, uy], [[sigma**2, 0.0], [0.0, sigma**2]])

def classify1(presses):
    if len(presses) < 1:
        return ''

    lps = []
    for side, (x, y) in presses:
        #print dists1[side].keys(), x, y
        # Evaluate letter distributions
        results = [(key, dists1[side][key].pdf([x, y])) for key in dists1[side]]
        
        #print [l for l, p in sorted(results, key = lambda x : x[1], reverse = True)]
        #print '-----'

        totalP = sum([p for l, p in results])

        # For each press, save dictionary matching probability of each letter
        lps.append(dict([(l, -numpy.log(p / totalP)) for l, p in results]))

    tws = []
    for w in wbL[len(presses)]:
        total = 0.0
        for i in range(len(w)):
            if w[i] in lps[i]:
                total += lps[i][w[i]]  # + -numpy.log(lprobs[l])
            else:
                total += 50000.0

        tws.append((w, total))# - numpy.log(wprobs[len(presses)][w])

    tws = sorted(tws, key = lambda x : x[1])

    return tws[0][0]#[w for w, p in tws[0:10]]
lhaxis = 0
lvaxis = 1

rhaxis = 3
rvaxis = 4

# Define some colors
BLACK    = (   0,   0,   0)
WHITE    = ( 255, 255, 255)

# This is a simple class that will help us print to the screen
# It has nothing to do with the joysticks, just outputing the
# information.
class TextPrint:
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def prints(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, [self.x, self.y])
        self.y += self.line_height
        
    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15
        
    def indent(self):
        self.x += 10
        
    def unindent(self):
        self.x -= 10
    

pygame.init()
 
# Set the width and height of the screen [width,height]
size = [600, 500]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("My Game")

#Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# Initialize the joysticks
pygame.joystick.init()
    
# Get ready to print
textPrint = TextPrint()

outLZone = False
outRZone = False
bbuttonOld = False

lxs = []
lys = []

presses = []
words = []
wordsByPresses = []

def findPeakOfArc(arc):
    points = numpy.array([[0.0, 0.0]] + arc + [[0.0, 0.0]])
    incremental_distances = [0.0] + [numpy.linalg.norm(p1 - p0) for p0, p1 in zip(points[:-1], points[1:])]
    distances = numpy.cumsum(incremental_distances)
    idx = bisect.bisect_left(distances, distances[-1] / 2)
    
    dx = distances[idx + 1] - distances[idx]
    if dx > 0:
        alpha = 1.0 - (distances[-1] / 2 - distances[idx]) / dx
        coords = points[idx] * alpha + points[idx + 1] * (1 - alpha)
    else:
        coords = points[idx]
    
    return coords
    
# -------- Main Program Loop -----------
while done==False:
    # EVENT PROCESSING STEP
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
            done = True # Flag that we are done so we exit this loop
    
    screen.fill(WHITE)
    textPrint.reset()

    # Get count of joysticks
    joystick_count = pygame.joystick.get_count()
    
    if joystick_count < 1:
        raise Exception("No joysticks detected")
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    name = joystick.get_name()
    textPrint.prints(screen, "Joystick name: {}".format(name) )
        
    # Usually axis run in pairs, up/down for one, and left/right for
    # the other.
    
    lx = [joystick.get_axis(lhaxis), joystick.get_axis(lvaxis)]
    rx = [joystick.get_axis(rhaxis), joystick.get_axis(rvaxis)]
    
    bbutton = bool(joystick.get_button(1))
    
    textPrint.prints(screen, "Left stick value: x {:>6.3f} y {:>6.3f}".format(lx[0], lx[1]))
    textPrint.prints(screen, "Right stick value: x {:>6.3f} y {:>6.3f}".format(rx[0], rx[1]))
    
    lr = numpy.linalg.norm(lx)
    rr = numpy.linalg.norm(rx)
    
    if lr > 0.25:
        if outLZone == False:
            lxs = []
            outLZone = True
        else:
            lxs.append(lx)
    else:
        if outLZone == True:
            presses.append((0, findPeakOfArc(lxs)))
            outLZone = False
            
    if rr > 0.25:
        if outRZone == False:
            rxs = []
            outRZone = True
        else:
            rxs.append(rx)
    else:
        if outRZone == True:
            loc = findPeakOfArc(rxs)
            if loc[0] > 0.5 and loc[1] > 0.5:
                words.append(classify1(presses))
                wordsByPresses.append(presses)
                presses = []
            else:
                presses.append((1, loc))
            outRZone = False
            
    if bbuttonOld == False and bbutton == True:
        bbuttonOld = True
        
    if bbuttonOld == True and bbutton == False:
        bbuttonOld = False
        
        if len(presses) > 0:
            presses.pop()
        elif len(wordsByPresses) > 0:
            presses = wordsByPresses.pop()
            words.pop()
            
    temp = classify1(presses)
    print '"{}"'.format(temp)
    textPrint.indent()
    textPrint.prints(screen, ' '.join(words + [temp]))
    textPrint.unindent()
    
    if True:
        line = ' '.join(words + [classify1(presses)])
        sys.stderr.write('\r')
        sys.stderr.write(' ' * (len(line) + 2))
        sys.stderr.write('\r')
        sys.stderr.write(line)
        sys.stderr.flush()

    # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT
    
    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # Limit to 20 frames per second
    clock.tick(60)
    
# Close the window and quit.
# If you forget this line, the program will 'hang'
# on exit if running from IDLE.
pygame.quit ()
