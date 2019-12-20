#!/usr/bin/env python
# coding: utf-8

# # Feature selection

# ## Identifying areas for feature selection
# Take an exploratory look at the post-feature engineering hiking dataset. Which of the following columns is a good candidate for feature selection?

# ### init: 1 dataframe

# In[2]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(hiking)
tobedownloaded="{pandas.core.frame.DataFrame: {'hiking.csv': 'https://file.io/xVUARo'}}"
prefix='data_from_datacamp/Chap4-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[3]:


import pandas as pd
hiking=pd.read_csv(prefix+'hiking.csv',index_col=0)


# ### code

# In[4]:


hiking.info()


# # Removing redundant features
# 

# ## Selecting relevant features
# Now that you've identified redundant columns in the volunteer dataset, let's perform feature selection on the dataset to return a DataFrame of the relevant features.

# ### init: 1 dataframe

# In[5]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(volunteer)
tobedownloaded="{pandas.core.frame.DataFrame: {'volunteer.csv': 'https://file.io/CA5102'}}"
prefix='data_from_datacamp/Chap4-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[6]:


import pandas as pd
volunteer=pd.read_csv(prefix+'volunteer.csv',index_col=0)


# ### code

# - Create a list of redundant column names and store it in the to_drop variable, in alphabetical order. You'll see three related features: locality, region, and postalcode. For now, let's only keep postalcode.
# - Drop the columns from the dataset using drop().
# - Print out the head() of the DataFrame to see the selected columns.

# In[7]:


volunteer.columns


# In[10]:


# Create a list of redundant column names to drop
to_drop = ["category_desc","created_date","locality", "region",   "vol_requests"]

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis=1)

# Print out the head of the new dataset
print(volunteer_subset.head())


# ## Checking for correlated features
# Let's take a look at the wine dataset again, which is made up of continuous, numerical features. Run Pearson's correlation coefficient on the dataset to determine which columns are good candidates for eliminating. Then, remove those columns from the DataFrame.

# ### init: 1 dataframe

# In[11]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(wine)
tobedownloaded="{pandas.core.frame.DataFrame: {'wine.csv': 'https://file.io/GzG4kD'}}"
prefix='data_from_datacamp/Chap4-Exercise2.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[12]:


import pandas as pd
wine=pd.read_csv(prefix+'wine.csv',index_col=0)


# ### code

# In[14]:


# Print out the column correlations of the wine dataset
print(wine.corr())

# Take a minute to find the column where the correlation value is greater than 0.75 at least twice
to_drop = "Flavanoids"

# Drop that column from the DataFrame
wine = wine.drop(to_drop, axis=1)


# # Selecting features using text vectors
# 

# ## Exploring text vectors, part 1
# Let's expand on the text vector exploration method we just learned about, using the volunteer dataset's title tf/idf vectors. In this first part of text vector exploration, we're going to add to that function we learned about in the slides. We'll return a list of numbers with the function. In the next exercise, we'll write another function to collect the top words across all documents, extract them, and then use that list to filter down our text_tfidf vector.

# ### init: 1 dataframe

# In[15]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(volunteer)
tobedownloaded="{pandas.core.frame.DataFrame: {'volunteer.csv': 'https://file.io/r6U231'}}"
prefix='data_from_datacamp/Chap4-Exercise3.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[16]:


import pandas as pd
volunteer=pd.read_csv(prefix+'volunteer.csv',index_col=0)


# In[24]:


vocab={1048: 'web', 278: 'designer', 1017: 'urban', 38: 'adventures', 490: 'ice', 890: 'skating', 90: 'at', 559: 'lasker', 832: 'rink', 368: 'fight', 423: 'global', 487: 'hunger', 68: 'and', 944: 'support', 1061: 'women', 356: 'farmers', 535: 'join', 969: 'the', 708: 'oxfam', 27: 'action', 240: 'corps', 498: 'in', 680: 'nyc', 922: 'stop', 947: 'swap', 790: 'queens', 911: 'staff', 281: 'development', 992: 'trainer', 200: 'claro', 145: 'brooklyn', 1037: 'volunteer', 93: 'attorney', 221: 'community', 455: 'health', 43: 'advocates', 942: 'supervise', 189: 'children', 466: 'highland', 717: 'park', 409: 'garden', 1071: 'worldofmoney', 696: 'org', 1085: 'youth', 60: 'amazing', 791: 'race', 789: 'qualified', 133: 'board', 620: 'member', 860: 'seats', 98: 'available', 1083: 'young', 33: 'adult', 1006: 'tutor', 1016: 'updated', 11: '30', 0: '11', 513: 'insurance', 199: 'claims', 600: 'manager', 979: 'timebanksnyc', 432: 'great', 340: 'exchange', 205: 'clean', 1015: 'up', 81: 'asbury', 171: 'cementary', 918: 'staten', 524: 'island', 869: 'senior', 194: 'citizen', 392: 'friendly', 1033: 'visitor', 881: 'shop', 1000: 'tree', 161: 'care', 1068: 'workshop', 4: '20', 646: 'movie', 856: 'screener', 380: 'for', 870: 'seniors', 355: 'farm', 430: 'graphic', 691: 'open', 480: 'house', 416: 'get', 984: 'tools', 980: 'to', 806: 'recycling', 1039: 'volunteers', 660: 'needed', 353: 'family', 336: 'event', 207: 'clerical', 158: 'cancer', 1041: 'walk', 120: 'befitnyc', 739: 'physical', 30: 'activity', 700: 'organizers', 269: 'decision', 266: 'day', 5: '2011', 661: 'needs', 1084: 'your', 459: 'help', 405: 'gain', 1021: 'valuable', 245: 'counseling', 344: 'experience', 687: 'on', 845: 'samaritans', 9: '24', 479: 'hour', 255: 'crisis', 478: 'hotline', 457: 'heart', 407: 'gallery', 703: 'our', 503: 'info', 949: 'table', 373: 'finding', 471: 'homes', 542: 'kids', 1077: 'yiddish', 903: 'speaking', 472: 'homework', 460: 'helper', 892: 'skilled', 800: 'rebuilding', 982: 'together', 468: 'home', 818: 'repairs', 438: 'greenteam', 40: 'advetures', 940: 'summer', 931: 'streets', 1005: 'tuesday', 335: 'evenings', 1060: 'with', 612: 'masa', 594: 'lunch', 770: 'program', 1018: 'us', 706: 'outreach', 618: 'meals', 760: 'preparedness', 222: 'compost', 773: 'project', 613: 'master', 223: 'composter', 178: 'certificate', 249: 'course', 318: 'emblemhealth', 144: 'bronx', 683: 'of', 873: 'service', 531: 'jcc', 601: 'manhattan', 418: 'girl', 855: 'scout', 872: 'series', 296: 'dorot', 838: 'rosh', 452: 'hashanah', 709: 'package', 274: 'delivery', 713: 'painting', 511: 'instructor', 530: 'jasa', 464: 'hes', 172: 'center', 12: '3rd', 70: 'annual', 377: 'flyny', 548: 'kite', 366: 'festival', 983: 'tomorrow', 151: 'business', 566: 'leaders', 955: 'teach', 110: 'basics', 465: 'high', 852: 'schoolers', 410: 'gardening', 397: 'ft', 1004: 'tryon', 910: 'st', 610: 'martin', 748: 'poetry', 668: 'new', 1079: 'york', 216: 'college', 424: 'goal', 941: 'sunday', 361: 'february', 6: '2012', 262: 'dance', 8: '22nd', 560: 'latino', 604: 'march', 2: '17', 1013: 'university', 848: 'saturday', 1008: 'tutors', 744: 'planet', 485: 'human', 602: 'mapping', 420: 'give', 1050: 'week', 186: 'child', 569: 'learn', 796: 'read', 926: 'storytelling', 243: 'costume', 597: 'making', 912: 'stage', 277: 'design', 319: 'emergency', 351: 'fair', 17: '9th', 1053: 'west', 887: 'side', 248: 'county', 676: 'nutrition', 314: 'educator', 879: 'shape', 306: 'east', 13: '54st', 801: 'rec', 1046: 'water', 45: 'aerobics', 83: 'asser', 573: 'levy', 712: 'paint', 57: 'alongside', 783: 'publicolor', 936: 'students', 536: 'jumpstart', 797: 'readers', 564: 'lead', 252: 'crafts', 408: 'games', 348: 'face', 751: 'popcorn', 527: 'jackie', 835: 'robinson', 716: 'parent', 375: 'fitness', 916: 'starrett', 197: 'city', 585: 'line', 263: 'dancer', 615: 'math', 587: 'literacy', 114: 'be', 209: 'climb', 985: 'top', 608: 'marketing', 86: 'assistant', 313: 'education', 673: 'nonprofit', 867: 'seeks', 805: 'recruitment', 626: 'mentors', 810: 'register', 92: 'attend', 142: 'breakfast', 701: 'orientation', 529: 'january', 272: 'deliver', 1058: 'winter', 1031: 'visit', 65: 'an', 525: 'isolated', 342: 'exercise', 213: 'coach', 670: 'night', 115: 'beach', 180: 'change', 77: 'art', 772: 'programs', 229: 'consumer', 779: 'protection', 562: 'law', 589: 'liver', 579: 'life', 565: 'leader', 901: 'soup', 547: 'kitchen', 307: 'eastern', 534: 'john', 650: 'muir', 930: 'street', 1024: 'vendor', 641: 'monthly', 959: 'team', 367: 'fiesta', 977: 'throgs', 658: 'neck', 224: 'computer', 956: 'teacher', 567: 'leadership', 244: 'council', 693: 'opportunity', 231: 'conversation', 461: 'helpers', 427: 'grades', 714: 'pantry', 288: 'distribution', 305: 'earth', 960: 'tech', 1049: 'website', 692: 'opportunities', 175: 'cents', 19: 'ability', 203: 'classroom', 877: 'set', 146: 'brush', 545: 'kindness', 999: 'transportation', 58: 'alternatives', 129: 'bike', 1020: 'valet', 1026: 'video', 311: 'editing', 767: 'professionals', 921: 'stipend', 49: 'after', 851: 'school', 624: 'mentor', 666: 'networking', 138: 'bowling', 398: 'fun', 449: 'harlem', 555: 'lanes', 866: 'seeking', 1078: 'yoga', 902: 'spanish', 695: 'or', 389: 'french', 362: 'feed', 488: 'hungry', 1080: 'yorkers', 14: '55', 690: 'only', 735: 'phone', 106: 'bank', 819: 'representative', 795: 'reach', 704: 'out', 643: 'morris', 458: 'heights', 904: 'special', 155: 'camp', 946: 'susan', 551: 'komen', 259: 'cure', 433: 'greater', 47: 'affiliate', 303: 'dumbo', 79: 'arts', 698: 'organizational', 148: 'budget', 639: 'money', 596: 'makes', 871: 'sense', 994: 'training', 889: 'site', 1027: 'videographer', 376: 'fly', 152: 'by', 970: 'theater', 429: 'grant', 1074: 'writer', 745: 'planning', 778: 'proposal', 759: 'preparation', 399: 'fund', 793: 'raising', 450: 'harm', 808: 'reduction', 35: 'adv', 515: 'intern', 875: 'serving', 575: 'lgbt', 34: 'adults', 482: 'how', 830: 'ride', 130: 'bikes', 821: 'research', 401: 'fundraising', 280: 'developement', 233: 'cook', 840: 'row', 50: 'afterschool', 630: 'middle', 885: 'shower', 400: 'fundraisers', 526: 'it', 519: 'interpreters', 563: 'lawyers', 446: 'haitian', 18: 'abe', 757: 'pre', 412: 'ged', 640: 'monitor', 89: 'astoria', 634: 'million', 1001: 'trees', 421: 'giveaway', 290: 'do', 1081: 'you', 1044: 'want', 595: 'make', 283: 'difference', 204: 'classwish', 896: 'snow', 883: 'shoveling', 196: 'citizenship', 761: 'press', 586: 'list', 781: 'public', 813: 'relations', 743: 'plan', 829: 'review', 394: 'friendship', 753: 'positive', 121: 'beginnings', 546: 'kit', 611: 'mary', 803: 'recreation', 291: 'does', 697: 'organization', 659: 'need', 858: 'search', 928: 'strategy', 332: 'esl', 46: 'affected', 924: 'storm', 995: 'transform', 590: 'lives', 933: 'strengthen', 220: 'communities', 119: 'become', 302: 'driver', 1025: 'veterans', 191: 'chinese', 997: 'translator', 512: 'instructors', 653: 'museum', 621: 'membership', 275: 'department', 284: 'director', 117: 'beautify', 996: 'transitional', 822: 'residence', 470: 'homeless', 623: 'men', 953: 'tank', 517: 'internship', 774: 'projects', 841: 'run', 1056: 'wild', 139: 'boys', 475: 'hope', 419: 'girls', 219: 'communications', 792: 'raise', 100: 'awareness', 31: 'administrative', 56: 'alliance', 811: 'registrar', 647: 'ms', 1062: 'word', 162: 'career', 246: 'counselor', 722: 'passover', 304: 'early', 188: 'childhood', 149: 'build', 747: 'plastic', 137: 'bottle', 857: 'sculpture', 763: 'pride', 523: 'is', 538: 'just', 76: 'around', 238: 'corner', 520: 'involved', 675: 'now', 390: 'fresh', 53: 'air', 957: 'teachers', 372: 'find', 729: 'perfect', 533: 'job', 684: 'office', 1075: 'writing', 264: 'data', 326: 'entry', 29: 'activism', 738: 'photography', 843: 'salesforce', 265: 'database', 261: 'customization', 736: 'photo', 333: 'essay', 572: 'legal', 42: 'advisor', 467: 'hike', 974: 'thon', 236: 'coordinator', 558: 'laser', 950: 'tag', 298: 'dowling', 3: '175th', 505: 'information', 962: 'technology', 352: 'fall', 382: 'forest', 826: 'restoration', 541: 'kickoff', 1002: 'trevor', 582: 'lifeline', 247: 'counselors', 973: 'thomas', 532: 'jefferson', 614: 'materials', 1076: 'year', 386: 'founder', 341: 'executive', 453: 'haunted', 557: 'lantern', 989: 'tours', 383: 'fort', 986: 'totten', 657: 'national', 878: 'sexual', 82: 'assault', 689: 'online', 993: 'trainers', 48: 'african', 63: 'american', 210: 'clothing', 301: 'drive', 828: 'returning', 865: 'seeds', 939: 'success', 746: 'plant', 981: 'today', 443: 'growth', 1009: 'udec', 328: 'enviromedia', 636: 'mobile', 606: 'maritime', 102: 'bacchanal', 742: 'pirates', 365: 'fest', 492: 'ikea', 329: 'erie', 111: 'basin', 282: 'diabetes', 88: 'association', 364: 'feria', 267: 'de', 844: 'salud', 664: 'nepali', 105: 'bangla', 784: 'punjabi', 998: 'translators', 674: 'not', 769: 'profit', 741: 'pioneer', 159: 'capoeira', 1023: 'various', 752: 'positions', 287: 'dispatcher', 991: 'trainee', 506: 'ing', 603: 'marathon', 388: 'free', 593: 'love', 135: 'books', 268: 'dear', 96: 'authors', 52: 'aide', 850: 'scheuer', 627: 'merchandise', 293: 'donate', 943: 'supplies', 360: 'feast', 406: 'gala', 112: 'battery', 833: 'rise', 919: 'stay', 787: 'put', 820: 'rescue', 897: 'soccer', 402: 'futsal', 730: 'performing', 36: 'advanced', 202: 'classes', 1070: 'world', 854: 'science', 1054: 'western', 64: 'americorps', 25: 'aces', 310: 'economic', 864: 'security', 507: 'initiative', 331: 'esi', 633: 'mill', 173: 'centers', 631: 'midtown', 1088: 'zumba', 1030: 'vision', 635: 'mission', 66: 'analysis', 552: 'lab', 958: 'teaching', 84: 'assist', 827: 'resume', 150: 'building', 899: 'society', 214: 'coaches', 1040: 'vs', 218: 'committee', 842: 'russian', 385: 'foster', 170: 'celebration', 616: 'may', 7: '21th', 688: 'one', 711: 'pager', 294: 'donation', 489: 'hurricane', 521: 'irene', 354: 'far', 836: 'rockaway', 325: 'enjoy', 1066: 'working', 686: 'olympics', 988: 'tournament', 798: 'reading', 719: 'partners', 234: 'cooper', 909: 'square', 975: 'thrift', 908: 'spring', 166: 'case', 599: 'management', 404: 'fvcp', 990: 'trail', 254: 'crew', 447: 'halloween', 165: 'carnival', 1042: 'walkathon', 359: 'feasibility', 67: 'analyst', 749: 'police', 868: 'seminar', 1064: 'work', 1035: 'visually', 496: 'impaired', 964: 'teens', 972: 'this', 322: 'energy', 315: 'efficiency', 321: 'end', 859: 'season', 156: 'campaign', 123: 'benefits', 802: 'reception', 300: 'drill', 237: 'copywriting', 235: 'coord', 454: 'have', 725: 'penchant', 55: 'all', 971: 'things', 1028: 'vintage', 976: 'thriftshop', 718: 'partner', 726: 'pencil', 720: 'partnership', 710: 'packing', 16: '8th', 907: 'sports', 346: 'expo', 164: 'cares', 184: 'cheerleaders', 1045: 'wanted', 445: 'habitat', 371: 'finance', 215: 'coffee', 324: 'english', 755: 'practice', 570: 'learners', 456: 'healthy', 28: 'active', 978: 'time', 122: 'benefit', 73: 'april', 357: 'fashion', 929: 'strawberry', 87: 'assistants', 174: 'central', 1087: 'zoo', 1: '125th', 127: 'bideawee', 440: 'greeters', 592: 'looking', 799: 'real', 495: 'impact', 504: 'inform', 728: 'people', 756: 'practices', 580: 'lifebeat', 413: 'general', 932: 'streetsquash', 286: 'discovery', 874: 'services', 663: 'neighborhood', 768: 'profiles', 951: 'take', 915: 'stand', 51: 'against', 1029: 'violence', 345: 'expert', 41: 'advice', 537: 'june', 849: 'schedule', 258: 'crowdfunding', 727: 'penny', 451: 'harvest', 434: 'green', 185: 'chefs', 677: 'nutritionists', 379: 'foodies', 625: 'mentoring', 136: 'boom', 669: 'newsletter', 217: 'come', 934: 'strides', 1043: 'walks', 187: 'childcare', 898: 'social', 619: 'media', 422: 'giving', 157: 'can', 61: 'ambassador', 10: '2nd', 967: 'thanksgiving', 363: 'feeding', 662: 'needy', 782: 'publicity', 723: 'patient', 163: 'caregiver', 1032: 'visiting', 469: 'homebound', 358: 'fc', 679: 'nyawc', 384: 'forum', 21: 'about', 1038: 'volunteering', 809: 'refreshments', 847: 'sara', 837: 'roosevelt', 206: 'cleanup', 116: 'beautification', 337: 'events', 69: 'animal', 484: 'hudson', 834: 'river', 605: 'mariners', 825: 'response', 343: 'exhibit', 20: 'aboard', 584: 'lilac', 208: 'client', 1052: 'welcome', 279: 'desk', 685: 'older', 574: 'lexington', 251: 'craft', 750: 'poll', 1065: 'workers', 518: 'interperters', 24: 'accounting', 85: 'assistance', 477: 'hosting', 776: 'promotion', 1011: 'unicef', 954: 'tap', 814: 'release', 270: 'dedication', 771: 'programming', 500: 'incarnation', 295: 'donor', 544: 'kieran', 906: 'sponsorship', 1069: 'workshops', 118: 'because', 338: 'every', 276: 'deserves', 179: 'chance', 740: 'pin', 273: 'delivered', 886: 'shred', 15: '5th', 99: 'avenue', 169: 'cdsc', 917: 'starving', 78: 'artist', 884: 'show', 948: 'system', 396: 'front', 880: 'share', 553: 'lanch', 935: 'student', 463: 'hemophilia', 577: 'liason', 629: 'methodist', 476: 'hospital', 113: 'bay', 831: 'ridge', 124: 'benonhurst', 75: 'area', 900: 'sought', 97: 'autistic', 297: 'douglaston', 788: 'qns', 812: 'registration', 32: 'administrator', 153: 'call', 426: 'governor', 804: 'recruiter', 786: 'purim', 327: 'envelope', 938: 'stuffing', 528: 'jam', 462: 'helpline', 923: 'store', 374: 'first', 415: 'generation', 1022: 'van', 241: 'cortlandt', 816: 'remembrance', 945: 'survey', 823: 'resonations', 143: 'breast', 323: 'engine', 694: 'optimization', 622: 'memorial', 894: 'sloan', 540: 'kettering', 435: 'greenhouse', 436: 'greening', 227: 'concert', 334: 'evacuation', 824: 'resources', 417: 'gift', 126: 'bicycling', 656: 'my', 393: 'friends', 473: 'honor', 1051: 'weekend', 731: 'person', 651: 'mural', 312: 'editor', 732: 'personal', 882: 'shopper', 764: 'pro', 134: 'bono', 253: 'create', 160: 'cards', 920: 'step', 672: 'non', 780: 'provider', 516: 'interns', 645: 'motion', 431: 'graphics', 125: 'best', 147: 'buddies', 502: 'inern', 103: 'back', 588: 'little', 242: 'cosmetologist', 107: 'barber', 1036: 'vocational', 72: 'apartment', 439: 'greeter', 766: 'professional', 1019: 'use', 893: 'skills', 702: 'others', 369: 'figure', 257: 'croton', 190: 'chinatown', 193: 'ci', 758: 'prep', 239: 'corporate', 1063: 'wordpress', 132: 'blog', 510: 'instructer', 807: 'red', 474: 'hook', 289: 'divert', 966: 'textiles', 395: 'from', 554: 'landfill', 437: 'greenmarket', 965: 'textile', 154: 'calling', 195: 'citizens', 497: 'improve', 26: 'achievement', 721: 'passion', 481: 'housing', 1067: 'works', 499: 'inc', 441: 'group', 299: 'drama', 561: 'laundromats', 320: 'employment', 927: 'strategic', 667: 'never', 104: 'bad', 391: 'friend', 403: 'future', 201: 'class', 1059: 'wish', 387: 'fpcj', 1072: 'worship', 1010: 'undergraduate', 428: 'graduate', 228: 'conference', 1047: 'we', 775: 'promote', 550: 'knowledge', 715: 'parade', 74: 'archivist', 425: 'google', 44: 'adwords', 493: 'imentor', 642: 'more', 598: 'male', 632: 'miles', 637: 'moms', 183: 'charity', 176: 'century', 987: 'tour', 198: 'civil', 724: 'patrol', 62: 'america', 539: 'kept', 862: 'secret', 648: 'ms131', 549: 'knitter', 256: 'crochet', 131: 'blankets', 177: 'ceo', 591: 'logo', 1012: 'unique', 1057: 'will', 128: 'big', 37: 'adventure', 23: 'accountant', 876: 'session', 888: 'single', 644: 'mothers', 192: 'choice', 895: 'smc', 1055: 'wii', 705: 'outdoor', 671: 'nights', 607: 'market', 514: 'intake', 638: 'monday', 141: 'branding', 140: 'brand', 491: 'identity', 649: 'mt', 1086: 'zion', 543: 'kidz', 817: 'reorganize', 578: 'library', 378: 'food', 91: 'athletic', 568: 'league', 655: 'musician', 59: 'alzheimer', 654: 'music', 109: 'bash', 765: 'proctor', 952: 'taking', 339: 'exams', 777: 'promotional', 733: 'personnel', 95: 'august', 891: 'skill', 665: 'networker', 309: 'ecological', 785: 'puppet', 501: 'income', 414: 'generating', 699: 'organizations', 250: 'cpr', 576: 'lgbtq', 317: 'el', 652: 'museo', 271: 'del', 108: 'barrio', 628: 'met', 330: 'escort', 846: 'sand', 167: 'castle', 230: 'contest', 853: 'schools', 486: 'humanities', 80: 'as', 861: 'second', 556: 'language', 101: 'babies', 963: 'teen', 54: 'al', 682: 'oerter', 483: 'html', 260: 'curriculum', 737: 'photographer', 863: 'secretary', 754: 'pr', 1073: 'would', 583: 'like', 225: 'computers', 961: 'technical', 442: 'grownyc', 968: 'that', 347: 'extraordinary', 381: 'foreclosure', 762: 'prevention', 681: 'nylag', 678: 'ny', 226: 'concern', 509: 'inspire', 22: 'academic', 1007: 'tutoring', 794: 'rbi', 71: 'anyone', 211: 'cma', 212: 'cms', 232: 'conversion', 308: 'eating', 571: 'learning', 181: 'chaperones', 1034: 'visits', 411: 'gear', 1014: 'unlimited', 581: 'lifeguard', 350: 'facilitators', 1003: 'troop', 839: 'route', 609: 'marshall', 508: 'inmotion', 925: 'story', 913: 'stair', 292: 'domestic', 168: 'catskills', 815: 'relief', 316: 'effort', 94: 'audience', 734: 'pharmacy', 444: 'guide', 707: 'overnight', 494: 'immediate', 285: 'dirty', 448: 'hands', 349: 'facilitator', 905: 'specialist', 182: 'chapter', 914: 'stamps', 522: 'iridescent', 937: 'studio', 39: 'advertising', 370: 'filmmakers', 617: 'mayor', 1082: 'youcantoo'}

# Take the title text
title_text = volunteer['title']
from sklearn.feature_extraction.text import TfidfVectorizer
# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
tfidf_vec.fit_transform(title_text)

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)


# ### code

# - Add parameters called original_vocab, for the tfidf_vec.vocabulary_, and top_n.
# - Call pd.Series on the zipped dictionary. This will make it easier to operate on.
# - Use the sort_values function to sort the series and slice the index up to top_n words.
# - Call the function, setting original_vocab=tfidf_vec.vocabulary_, setting vector_index=8 to grab the 9th row, and setting top_n=3, to grab the top 3 weighted words.

# In[25]:


# Add in the rest of the parameters
def return_weights(vocab, original_vocab, vector, vector_index,  top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    
    # Let's transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    
    # Let's sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

# Print out the weighted words
print(return_weights(vocab, tfidf_vec.vocabulary_, text_tfidf, 8 , 3))


# ## Exploring text vectors, part 2
# Using the function we wrote in the previous exercise, we're going to extract the top words from each document in the text vector, return a list of the word indices, and use that list to filter the text vector down to those top words.

# ### code

# - Call return_weights to return the top weighted words for that document.
# - Call set on the returned filter_list so we don't get duplicated numbers.
# - Call words_to_filter, passing in the following parameters: vocab for the vocab parameter, tfidf_vec.vocabulary_ for the original_vocab parameter, text_tfidf for the vector parameter, and 3 to grab the top_n 3 weighted words from each document.
# - Finally, pass that filtered_words set into a list to use as a filter for the text vector.

# In[26]:


def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
    
        # Here we'll call the function from the previous exercise, and extend the list we're creating
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)

# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, 3)

# By converting filtered_words back to a list, we can use it to filter the columns in the text vector
filtered_text = text_tfidf[:, list(filtered_words)]


# ## Training Naive Bayes with feature selection
# Let's re-run the Naive Bayes text classification model we ran at the end of chapter 3, with our selection choices from the previous exercise, on the volunteer dataset's title and category_desc columns.
# 
# 

# ### code

# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()


# - Use train_test_split on the filtered_text text vector, the y labels (which is the category_desc labels), and pass the y set to the stratify parameter, since we have an uneven class distribution.
# - Fit the nb Naive Bayes model to train_X and train_y.
# - Score the nb model on the test_X and test_y test sets.
# 

# In[33]:


# Split the dataset according to the class distribution of category_desc, using the filtered_text vector
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), volunteer["category_desc"], stratify=volunteer["category_desc"])

# Fit the model to the training data
nb.fit(train_X, train_y)

# Print out the model's accuracy
print(nb.score(test_X, test_y))


# # Dimensionality reduction
# 

# ## Using PCA
# Let's apply PCA to the wine dataset, to see if we can get an increase in our model's accuracy.

# ### init: 1 dataframe

# In[35]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(wine)
tobedownloaded="{pandas.core.frame.DataFrame: {'wine.csv': 'https://file.io/NA7To2'}}"
prefix='data_from_datacamp/Chap4-Exercise4.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[36]:


import pandas as pd
wine=pd.read_csv(prefix+'wine.csv',index_col=0)


# ### code

# In[37]:


from sklearn.decomposition import PCA

# Set up PCA and the X vector for diminsionality reduction
pca = PCA()
wine_X = wine.drop("Type", axis=1)

# Apply PCA to the wine dataset X vector
transformed_X = pca.fit_transform(wine_X)

# Look at the percentage of variance explained by the different components
print(pca.explained_variance_ratio_)


# ## Training a model with PCA
# Now that we have run PCA on the wine dataset, let's try training a model with it.

# ### init

# In[39]:


y=wine[['Type']]


# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# ### code

# In[43]:


# Split the transformed X and the y labels into training and test sets
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(transformed_X, y)

# Fit knn to the training data
knn.fit(X_wine_train, y_wine_train)

# Score knn on the test data and print it out
print(knn.score(X_wine_test, y_wine_test))


# In[ ]:



