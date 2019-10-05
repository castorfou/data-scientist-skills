@echo on
set Path=C:\Users\F279814\AppData\Local\Continuum\anaconda3;C:\Users\F279814\AppData\Local\Continuum\anaconda3\Library\mingw-w64\bin;C:\Users\F279814\AppData\Local\Continuum\anaconda3\Library\usr\bin;C:\Users\F279814\AppData\Local\Continuum\anaconda3\Library\bin;C:\Users\F279814\AppData\Local\Continuum\anaconda3\Scripts;C:\Users\F279814\AppData\Local\Continuum\anaconda3\bin;C:\Users\F279814\AppData\Local\Continuum\anaconda3\condabin;%PATH%
REM open conda command prompt
d:
cd D:\git\datacamp-itp\data-scientist-skills\python-sandbox\pandas-access-db
call conda activate py37_32-access 
jupyter notebook