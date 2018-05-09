import os, time
import datetime

os.chdir('C:\\Sharekhan\\TradeTigerNew\\Chart')

for file in os.listdir():
    mtime = os.path.getmtime(file)
    yyyymmdd = datetime.datetime.fromtimestamp(mtime).strftime("%Y%m%d")
    
    if file[:8].isdigit():
        continue
    elif(datetime.datetime.now().strftime("%Y%m%d")==yyyymmdd):
        if (file[:2] == 'NC'):
            os.rename(file, file[2:].upper())
            print(file + " --> " + file[2:].upper())
        else:
            continue
    else:
        if (file[:2] == 'NC'):
            os.rename(file, yyyymmdd+'_'+file[2:].upper())
            print(file +' --> '+ yyyymmdd+'_'+file[2:].upper())
        else:
            os.rename(file, yyyymmdd+'_'+file.upper())
            print(file +' --> '+ yyyymmdd+'_'+file.upper())

    

