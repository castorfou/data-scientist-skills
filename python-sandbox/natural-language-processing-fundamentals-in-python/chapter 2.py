#!/usr/bin/env python
# coding: utf-8

# # Word counts with bag-of-words
# 

# ## Building a Counter with bag-of-words
# In this exercise, you'll build your first (in this course) bag-of-words counter using a Wikipedia article, which has been pre-loaded as article. Try doing the bag-of-words without looking at the full article text, and guessing what the topic is! If you'd like to peek at the title at the end, we've included it as article_title. Note that this article text has had very little preprocessing from the raw Wikipedia database entry.
# 
# word_tokenize has been imported for you.

# ### init

# In[1]:


article = '\'\'\'Debugging\'\'\' is the process of finding and resolving of defects that prevent correct operation of computer software or a system.  \n\nNumerous books have been written about debugging (see below: #Further reading|Further reading), as it involves numerous aspects, including interactive debugging, control flow, integration testing, Logfile|log files, monitoring (Application monitoring|application, System Monitoring|system), memory dumps, Profiling (computer programming)|profiling, Statistical Process Control, and special design tactics to improve detection while simplifying changes.\n\nOrigin\nA computer log entry from the Mark&nbsp;II, with a moth taped to the page\n\nThe terms "bug" and "debugging" are popularly attributed to Admiral Grace Hopper in the 1940s.[http://foldoc.org/Grace+Hopper Grace Hopper]  from FOLDOC While she was working on a Harvard Mark II|Mark II Computer at Harvard University, her associates discovered a moth stuck in a relay and thereby impeding operation, whereupon she remarked that they were "debugging" the system. However the term "bug" in the meaning of technical error dates back at least to 1878 and Thomas Edison (see software bug for a full discussion), and "debugging" seems to have been used as a term in aeronautics before entering the world of computers. Indeed, in an interview Grace Hopper remarked that she was not coining the term{{Citation needed|date=July 2015}}. The moth fit the already existing terminology, so it was saved.  A letter from J. Robert Oppenheimer (director of the WWII atomic bomb "Manhattan" project at Los Alamos, NM) used the term in a letter to Dr. Ernest Lawrence at UC Berkeley, dated October 27, 1944,http://bancroft.berkeley.edu/Exhibits/physics/images/bigscience25.jpg regarding the recruitment of additional technical staff.\n\nThe Oxford English Dictionary entry for "debug" quotes the term "debugging" used in reference to airplane engine testing in a 1945 article in the Journal of the Royal Aeronautical Society. An article in "Airforce" (June 1945 p.&nbsp;50) also refers to debugging, this time of aircraft cameras.  Hopper\'s computer bug|bug was found on September 9, 1947. The term was not adopted by computer programmers until the early 1950s.\nThe seminal article by GillS. Gill, [http://www.jstor.org/stable/98663 The Diagnosis of Mistakes in Programmes on the EDSAC], Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, Vol. 206, No. 1087 (May 22, 1951), pp. 538-554 in 1951 is the earliest in-depth discussion of programming errors, but it does not use the term "bug" or "debugging".\nIn the Association for Computing Machinery|ACM\'s digital library, the term "debugging" is first used in three papers from 1952 ACM National Meetings.Robert V. D. Campbell, [http://portal.acm.org/citation.cfm?id=609784.609786 Evolution of automatic computation], Proceedings of the 1952 ACM national meeting (Pittsburgh), p 29-32, 1952.Alex Orden, [http://portal.acm.org/citation.cfm?id=609784.609793 Solution of systems of linear inequalities on a digital computer], Proceedings of the 1952 ACM national meeting (Pittsburgh), p. 91-95, 1952.Howard B. Demuth, John B. Jackson, Edmund Klein, N. Metropolis, Walter Orvedahl, James H. Richardson, [http://portal.acm.org/citation.cfm?id=800259.808982 MANIAC], Proceedings of the 1952 ACM national meeting (Toronto), p. 13-16 Two of the three use the term in quotation marks.\nBy 1963 "debugging" was a common enough term to be mentioned in passing without explanation on page 1 of the Compatible Time-Sharing System|CTSS manual.[http://www.bitsavers.org/pdf/mit/ctss/CTSS_ProgrammersGuide.pdf The Compatible Time-Sharing System], M.I.T. Press, 1963\n\nKidwell\'s article \'\'Stalking the Elusive Computer Bug\'\'Peggy Aldrich Kidwell, [http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?tp=&arnumber=728224&isnumber=15706 Stalking the Elusive Computer Bug], IEEE Annals of the History of Computing, 1998. discusses the etymology of "bug" and "debug" in greater detail.\n\nScope\nAs software and electronic systems have become generally more complex, the various common debugging techniques have expanded with more methods to detect anomalies, assess impact, and schedule software patches or full updates to a system. The words "anomaly" and "discrepancy" can be used, as being more neutral terms, to avoid the words "error" and "defect" or "bug" where there might be an implication that all so-called \'\'errors\'\', \'\'defects\'\' or \'\'bugs\'\' must be fixed (at all costs). Instead, an impact assessment can be made to determine if changes to remove an \'\'anomaly\'\' (or \'\'discrepancy\'\') would be cost-effective for the system, or perhaps a scheduled new release might render the change(s) unnecessary. Not all issues are life-critical or mission-critical in a system. Also, it is important to avoid the situation where a change might be more upsetting to users, long-term, than living with the known problem(s) (where the "cure would be worse than the disease"). Basing decisions of the acceptability of some anomalies can avoid a culture of a "zero-defects" mandate, where people might be tempted to deny the existence of problems so that the result would appear as zero \'\'defects\'\'. Considering the collateral issues, such as the cost-versus-benefit impact assessment, then broader debugging techniques will expand to determine the frequency of anomalies (how often the same "bugs" occur) to help assess their impact to the overall system.\n\nTools\nDebugging on video game consoles is usually done with special hardware such as this Xbox (console)|Xbox debug unit intended for developers.\n\nDebugging ranges in complexity from fixing simple errors to performing lengthy and tiresome tasks of data collection, analysis, and scheduling updates.  The debugging skill of the programmer can be a major factor in the ability to debug a problem, but the difficulty of software debugging varies greatly with the complexity of the system, and also depends, to some extent, on the programming language(s) used and the available tools, such as \'\'debuggers\'\'. Debuggers are software tools which enable the programmer to monitor the execution (computers)|execution of a program, stop it, restart it, set breakpoints, and change values in memory. The term \'\'debugger\'\' can also refer to the person who is doing the debugging.\n\nGenerally, high-level programming languages, such as Java (programming language)|Java, make debugging easier, because they have features such as exception handling that make real sources of erratic behaviour easier to spot. In programming languages such as C (programming language)|C or assembly language|assembly, bugs may cause silent problems such as memory corruption, and it is often difficult to see where the initial problem happened. In those cases, memory debugging|memory debugger tools may be needed.\n\nIn certain situations, general purpose software tools that are language specific in nature can be very useful.  These take the form of \'\'List of tools for static code analysis|static code analysis tools\'\'.  These tools look for a very specific set of known problems, some common and some rare, within the source code.  All such issues detected by these tools would rarely be picked up by a compiler or interpreter, thus they are not syntax checkers, but more semantic checkers.  Some tools claim to be able to detect 300+ unique problems. Both commercial and free tools exist in various languages.  These tools can be extremely useful when checking very large source trees, where it is impractical to do code walkthroughs.  A typical example of a problem detected would be a variable dereference that occurs \'\'before\'\' the variable is assigned a value.  Another example would be to perform strong type checking when the language does not require such.  Thus, they are better at locating likely errors, versus actual errors.  As a result, these tools have a reputation of false positives.  The old Unix \'\'Lint programming tool|lint\'\' program is an early example.\n\nFor debugging electronic hardware (e.g., computer hardware) as well as low-level software (e.g., BIOSes, device drivers) and firmware, instruments such as oscilloscopes, logic analyzers or in-circuit emulator|in-circuit emulators (ICEs) are often used, alone or in combination.  An ICE may perform many of the typical software debugger\'s tasks on low-level software and firmware.\n\nDebugging process \nNormally the first step in debugging is to attempt to reproduce the problem. This can be a non-trivial task, for example as with Parallel computing|parallel processes or some unusual software bugs. Also, specific user environment and usage history can make it difficult to reproduce the problem.\n\nAfter the bug is reproduced, the input of the program may need to be simplified to make it easier to debug. For example, a bug in a compiler can make it Crash (computing)|crash when parsing some large source file. However, after simplification of the test case, only few lines from the original source file can be sufficient to reproduce the same crash. Such simplification can be made manually, using a Divide and conquer algorithm|divide-and-conquer approach. The programmer will try to remove some parts of original test case and check if the problem still exists. When debugging the problem in a Graphical user interface|GUI, the programmer can try to skip some user interaction from the original problem description and check if remaining actions are sufficient for bugs to appear.\n\nAfter the test case is sufficiently simplified, a programmer can use a debugger tool to examine program states (values of variables, plus the call stack) and track down the origin of the problem(s). Alternatively, Tracing (software)|tracing can be used. In simple cases, tracing is just a few print statements, which output the values of variables at certain points of program execution.{{citation needed|date=February 2016}}\n\n Techniques \n \'\'Interactive debugging\'\'\n \'\'{{visible anchor|Print debugging}}\'\' (or tracing) is the act of watching (live or recorded) trace statements, or print statements, that indicate the flow of execution of a process. This is sometimes called \'\'{{visible anchor|printf debugging}}\'\', due to the use of the printf function in C. This kind of debugging was turned on by the command TRON in the original versions of the novice-oriented BASIC programming language. TRON stood for, "Trace On." TRON caused the line numbers of each BASIC command line to print as the program ran.\n \'\'Remote debugging\'\' is the process of debugging a program running on a system different from the debugger. To start remote debugging, a debugger connects to a remote system over a network. The debugger can then control the execution of the program on the remote system and retrieve information about its state.\n \'\'Post-mortem debugging\'\' is debugging of the program after it has already Crash (computing)|crashed. Related techniques often include various tracing techniques (for example,[http://www.drdobbs.com/tools/185300443 Postmortem Debugging, Stephen Wormuller, Dr. Dobbs Journal, 2006]) and/or analysis of memory dump (or core dump) of the crashed process. The dump of the process could be obtained automatically by the system (for example, when process has terminated due to an unhandled exception), or by a programmer-inserted instruction, or manually by the interactive user.\n \'\'"Wolf fence" algorithm:\'\' Edward Gauss described this simple but very useful and now famous algorithm in a 1982 article for communications of the ACM as follows: "There\'s one wolf in Alaska; how do you find it? First build a fence down the middle of the state, wait for the wolf to howl, determine which side of the fence it is on. Repeat process on that side only, until you get to the point where you can see the wolf."<ref name="communications of the ACM">{{cite journal | title="Pracniques: The "Wolf Fence" Algorithm for Debugging", | author=E. J. Gauss | year=1982}} This is implemented e.g. in the Git (software)|Git version control system as the command \'\'git bisect\'\', which uses the above algorithm to determine which Commit (data management)|commit introduced a particular bug.\n \'\'Delta Debugging\'\'{{snd}} a technique of automating test case simplification.Andreas Zeller: <cite>Why Programs Fail: A Guide to Systematic Debugging</cite>, Morgan Kaufmann, 2005. ISBN 1-55860-866-4{{rp|p.123}}<!-- for redirect from \'Saff Squeeze\' -->\n \'\'Saff Squeeze\'\'{{snd}} a technique of isolating failure within the test using progressive inlining of parts of the failing test.[http://www.threeriversinstitute.org/HitEmHighHitEmLow.html Kent Beck, Hit \'em High, Hit \'em Low: Regression Testing and the Saff Squeeze]\n\nDebugging for embedded systems\nIn contrast to the general purpose computer software design environment, a primary characteristic of embedded environments is the sheer number of different platforms available to the developers (CPU architectures, vendors, operating systems and their variants). Embedded systems are, by definition, not general-purpose designs: they are typically developed for a single task (or small range of tasks), and the platform is chosen specifically to optimize that application. Not only does this fact make life tough for embedded system developers, it also makes debugging and testing of these systems harder as well, since different debugging tools are needed in different platforms.\n\nto identify and fix bugs in the system (e.g. logical or synchronization problems in the code, or a design error in the hardware);\nto collect information about the operating states of the system that may then be used to analyze the system: to find ways to boost its performance or to optimize other important characteristics (e.g. energy consumption, reliability, real-time response etc.).\n\nAnti-debugging\nAnti-debugging is "the implementation of one or more techniques within computer code that hinders attempts at reverse engineering or debugging a target process".<ref name="veracode-antidebugging">{{cite web |url=http://www.veracode.com/blog/2008/12/anti-debugging-series-part-i/ |title=Anti-Debugging Series - Part I |last=Shields |first=Tyler |date=2008-12-02 |work=Veracode |accessdate=2009-03-17}} It is actively used by recognized publishers in copy protection|copy-protection schemas, but is also used by malware to complicate its detection and elimination.<ref name="soft-prot">[http://people.seas.harvard.edu/~mgagnon/software_protection_through_anti_debugging.pdf Software Protection through Anti-Debugging Michael N Gagnon, Stephen Taylor, Anup Ghosh] Techniques used in anti-debugging include:\nAPI-based: check for the existence of a debugger using system information\nException-based: check to see if exceptions are interfered with\nProcess and thread blocks: check whether process and thread blocks have been manipulated\nModified code: check for code modifications made by a debugger handling software breakpoints\nHardware- and register-based: check for hardware breakpoints and CPU registers\nTiming and latency: check the time taken for the execution of instructions\nDetecting and penalizing debugger<ref name="soft-prot" /><!-- reference does not exist -->\n\nAn early example of anti-debugging existed in early versions of Microsoft Word which, if a debugger was detected, produced a message that said: "The tree of evil bears bitter fruit. Now trashing program disk.", after which it caused the floppy disk drive to emit alarming noises with the intent of scaring the user away from attempting it again.<ref name="SecurityEngineeringRA">{{cite book | url=http://www.cl.cam.ac.uk/~rja14/book.html | author=Ross J. Anderson | title=Security Engineering | isbn = 0-471-38922-6 | page=684 }}<ref name="toastytech">{{cite web | url=http://toastytech.com/guis/word1153.html | title=Microsoft Word for DOS 1.15}}\n'


# In[2]:


article_title = 'Debugging'

from nltk.tokenize import sent_tokenize, word_tokenize


# ### code

# - Import Counter from collections.
# - Use word_tokenize() to split the article into tokens.
# - Use a list comprehension with t as the iterator variable to convert all the tokens into lowercase. The .lower() method converts text into lowercase.
# - Create a bag-of-words counter called bow_simple by using Counter() with lower_tokens as the argument.
# - Use the .most_common() method of bow_simple to print the 10 most common tokens.
# 

# In[3]:


# Import Counter
from collections import Counter

# Tokenize the article: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple.most_common(10))


# # Simple text preprocessing
# 

# ## Text preprocessing practice
# Now, it's your turn to apply the techniques you've learned to help clean up text for better NLP results. You'll need to remove stop words and non-alphabetic characters, lemmatize, and perform a new bag-of-words on your cleaned text.
# 
# You start with the same tokens you created in the last exercise: lower_tokens. You also have the Counter class imported.

# ### code

# - Import the WordNetLemmatizer class from nltk.stem.
# - Create a list alpha_only that contains only alphabetical characters. You can use the .isalpha() method to check for this.
# - Create another list called no_stops consisting of words from alpha_only that are not contained in english_stops.
# - Initialize a WordNetLemmatizer object called wordnet_lemmatizer and use its .lemmatize() method on the tokens in no_stops to create a new list called lemmatized.
# - Create a new Counter called bow with the lemmatized words.
# - Lastly, print the 10 most common tokens.
# 
# ![image.png](attachment:image.png)

# In[11]:


from nltk.corpus import stopwords
english_stops = stopwords.words('english')


# In[12]:


# Import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()]

# Remove all stop words: no_stops
no_stops = [t for t in alpha_only if t not in english_stops]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 10 most common tokens
print(bow.most_common(10))


# # Introduction to gensim
# 

# ## Creating and querying a corpus with gensim
# It's time to apply the methods you learned in the previous video to create your first gensim dictionary and corpus!
# 
# You'll use these data structures to investigate word trends and potential interesting topics in your document set. To get started, we have imported a few additional messy articles from Wikipedia, which were preprocessed by lowercasing all words, tokenizing them, and removing stop words and punctuation. These were then stored in a list of document tokens called articles. You'll need to do some light preprocessing and then generate the gensim dictionary and corpus.

# In[4]:


import gensim


# In[2]:


import sys, os
# */site-packages is where your current session is running its python out of
site_path = ''
for path in sys.path:
    if 'site-packages' in path.split('/')[-1]:
        print(path)
        site_path = path
# search to see if gensim in installed packages
if len(site_path) > 0:
    if not 'gensim' in os.listdir(site_path):
        print('package not found')
    else:
        print('gensim installed')    


# In[3]:


sys.path


# ### init

# In[3]:


from uploadfromdatacamp import saveFromFileIO

#artic_np = np.array(articles)
#uploadToFileIO(artic_np)
tobedownloaded="{numpy.ndarray: {'artic_np.csv': 'https://file.io/HZVBLF'}}"
prefix='data_from_datacamp/Chap2-Exercise3.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[8]:


from uploadfromdatacamp import loadNDArrayFromCsv

artic_np = loadNDArrayFromCsv(prefix+'artic_np.csv', dtype=None)


# In[17]:


myArray = np.fromfile(prefix+'artic_np.csv', sep=',', dtype='U')


# In[15]:


myArray


# In[ ]:




