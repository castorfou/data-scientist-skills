#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

data = pd.Series([1, 2, np.nan, 4, 5])
data.head()


# In[3]:


import numpy as np
import pandas as pd

data = pd.Series([1, 2, np.nan, 4, np.nan])
data.fillna(method = 'ffill')


# In[4]:


import numpy as np
import pandas as pd

data = pd.Series(['apple','banana','apple', 'banana'])
data.str.upper()


# In[8]:


from datetime import date

date_of_birth = date(1986, 12, 9)

fmt = '%b %d %Y'

print(date_of_birth.strftime(fmt))


# In[10]:


import pandas as pd

pd.DataFrame({'x':'a', 'x':'b'}, index = [0, 1])


# In[15]:


import pandas as pd

x = pd.Series(["2018-01-01", "2019-01-01"], dtype = "datetime64[ns]") 

x.dt.year.astype('int64')


# In[ ]:




