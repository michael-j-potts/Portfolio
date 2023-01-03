from bs4 import BeautifulSoup as bs
from bs4 import Comment
import wikipedia as wp
import requests
from datetime import datetime
import random
import shutil
import re
import time
import os

def DocDate(html, DateRegex):
    #Returns document write/update date in datetime format
    DateSentence = html.find(id = 'footer-info-lastmod').text
    DateUnprocessed = re.findall(DateRegex, DateSentence)
    for i in DateUnprocessed:
        Date = i
    Date = Date.split(' ')
    month = datetime.strptime(Date[1][0:3], "%b").month
    Date = datetime(int(Date[2]), int(month), int(Date[0]))
    return Date

def visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def DocBody(html):
    #Parses and returns the documents content
    Body = html.find(class_ = 'mw-parser-output')
    
    for element in Body(text = lambda text: isinstance(text, Comment)):
        element.extract()

    #blocked = ['style', 'script', 'head', 'title', 'meta', '[document]', {'class': 'toc'}, {'class': "reflist"}, {'class':"navbox"}]
    #content = [item for item in Body.find_all(text = True) if (item.parent.name not in blocked)]

    text = Body.find_all(text = True)
    content = filter(visible, text)
    
    #Reformat the text we gather to account from unnecessary additions.
    corrected = []
    corrected2 = []
    for i in content:
        corrected.append(i.replace('\xa0', ' '))
    for i in corrected:
        corrected2.append(re.sub("[\[].*?[\]]", "", i))
    Final = []    
    for i in corrected2:
        Final.append(i.strip())
    while("" in Final):
        Final.remove("")
    temp = []
    for i in Final:
        i = i.split(" ")
        for j in i:
            temp.append(j)
            temp.append(" ")
    Final = "".join(temp)
    return Final

def DocURLList(html, CurrentURL):
    #Records a list of all URLs found in the document body for PageRank and 
    #later for updating documents if they have since been edited.
    LinkList = []
    blocked = ["catlinks"]
    IgnoreList = ["/wiki/Main_Page", "/wiki/Category", "/wiki/Help", "/wiki/Wikipedia", "/wiki/Special", "/wiki/Portal", "/wiki/File", "/wiki/Talk",
    '/wiki/Template']
    content = [item for item in html.find_all(href = True) if (item.parent.name not in blocked)]
    if os.path.exists(Directory + "CoreURLs.txt"):
        Corelist = open((Directory + "CoreURLs.txt"), 'r').read().split('\n')
    for link in content:
        if link['href'][0:6] == "/wiki/":
            #Ignore other wiki links that are not articles
            if link['href'].startswith(tuple(IgnoreList)):
                pass
            #Avoid the current URL to avoid arriving at the Current URL again
            elif (RefPage + link['href'] == CurrentURL):
                pass
            #Avoid duplication in the current article
            else:
                if os.path.exists(Directory + "CoreURLs.txt"):
                    if ((RefPage + link['href'])) not in Corelist:
                        LinkList.append((RefPage + link['href']))
                else:
                    if ((RefPage + link['href'])) not in LinkList:
                        LinkList.append((RefPage + link['href']))
    URLList = "\n".join(LinkList)
    URLs = " ".join(LinkList)
    return URLList, URLs

def TrimPeripheralURLs(NumDocs):
    #Trims off URLs that are not found within the core
    #file, to account for graph edges only between present nodes in 
    #The graph subset.
    Core = open((Directory + "CoreURLs.txt"), 'r').read().split('\n')
    URLs = open((Directory + "IndividualURLList.txt"), 'r').read().split('\n')
    Trimmed = open((Directory + "Trimmed.txt"), 'a+')
    count = 0
    for i in URLs:
        for j in i.split(' '):
            if j in Core:
                Trimmed.write(j)
                Trimmed.write(' ')
        count += 1
        if count < NumDocs:
            Trimmed.write('\n')
    

#--Begin--
#Set some variables for starting locations/directories
RefPage = 'https://en.wikipedia.org'
RandomPage = "Https://en.wikipedia.org/wiki/Special:Random"
CorePages = int(input("How many core documents would you like to collect? (integer value only)\n"))
StartPreference = input("Do you have a topic you would like to start with? (y/n)\n")

Directory = './Repository/'
DateRegex = r"\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,4}"

if StartPreference == 'y':
    Query = input("What is your search query?")
    for i in wp.search(Query, results = 1):
        CurrentURL = (RefPage + '/wiki/' + str(i))
else:
    CurrentURL = RandomPage

#Deletes the repository if it exists for quick clean up before running the program
#a second time.
if os.path.exists(Directory):
    shutil.rmtree(Directory)
    
os.makedirs(Directory)
print("Created file repository")

#Set our first document as the user choice
try:
    document = requests.get(CurrentURL, timeout = 3)
except requests.exceptions.Timeout:
    print("Timeout, retrying...")
    document = requests.get(CurrentURL)

#Choose how many core documents you would like to gather.
TempURLlist = []
for n in range(CorePages):
    #Gather the documents html and soup it
    flag = False
    html = bs(document.content, "html.parser")

    #Call each of the pertinent functions to gather our data
    try:
        Title = html.find(class_ = "firstHeading").text
        if '/' in Title:
            Title = Title.replace('/', '-')
        if '.' in Title:
            Title = Title.replacec('.', '')
        PageURL = (html.find(rel = "canonical")['href'])
        TempURLlist.append(PageURL)
        print(PageURL)
        Date = DocDate(html, DateRegex)
        content = DocBody(html) 
        URLList, URLs = DocURLList(html, PageURL)
    except Exception:
        print("Wikipedia article not valid. Skipping.")
        flag = True
        pass

    if flag == False:
        #Write the data to a new/the respective text file
        filename = Directory + str(Title) + ".txt"
        file = open(filename, 'w')
        file.write(PageURL + '\n')
        file.write(Title + '\n')
        file.write(str(Date) + '\n')
        file.write(content + '\n')
        file.close()
    
        #Contains the respective URLs
        URLFile = open((Directory + "IndividualURLList.txt"), 'a+')
        URLFile.write(PageURL + ' ' + URLs + '\n')
        URLFile.write('')
        URLFile.close()

        #Contains the file names in order
        Namefile = open((Directory + "Names.txt"), 'a+')
        Namefile.write(Title + '\n')
        Namefile.close()

        #This contains only a list of all URLs with no separation
        #used for gathering all docs in a small subgraph.
        URLFile = open((Directory + "CoreURLs.txt"), 'a+')
        if os.path.getsize(Directory + 'CoreURLs.txt') > 0:
            URLFile.write('\n')
        URLFile.write(URLList)
        URLFile.write('')
        URLFile.close()

    #Completed and following webcrawler etiquette
    print("Completed scraping core file ", n+1, " out of ", len(range(CorePages)), ": ", Title)
    time.sleep(1.5)

    #choose the next document from one of the core URL list, to ensure at least a small
    #to large relation of document topics
    RemainingURLs = open((Directory + "CoreURLs.txt"), 'r')
    RemainingURLs = RemainingURLs.read().split('\n')
    CurrentURL = random.choice(RemainingURLs)
    try:
        document = requests.get(CurrentURL, timeout = 5)
    except requests.exceptions.Timeout:
        print("Timeout, retrying...")
        document = requests.get(CurrentURL)

#Once the core documents have been gathered, our program will explore the remaining unexplored
#URLs to gather their URL links. Only present file URLs (those in the original list, and 
#those the original list referenced) will be retained to make a small subgraph system.
RemainingURLs = open((Directory + "CoreURLs.txt"), 'r')
RemainingURLs = RemainingURLs.read().split('\n')

#remove the core files from the RemainingURL list to crawl
RemainingURLs = [x for x in RemainingURLs if x not in TempURLlist]

#Remove URLs that are already present in the URL list
UniqueURLs = list(set(RemainingURLs))

Peripheral = len(UniqueURLs)
count = 0
print(len(RemainingURLs))
print(Peripheral)
for i in UniqueURLs:
    flag = False
    #Gather the documents html and soup it
    try:
        document = requests.get(i, timeout = 5)
    except requests.exceptions.Timeout:
        print("Timeout, retrying...")
        document = requests.get(i)
    html = bs(document.content, "html.parser")

    #Call each of the pertinent functions to gather our data
    try:
        Title = html.find(class_ = "firstHeading").text
        if '/' in Title:
            Title = Title.replace('/', '-')
        if '.' in Title:
            Title = Title.replacec('.', '')
        PageURL = html.find(rel = "canonical")['href']
        print(PageURL)
        Date = DocDate(html, DateRegex)
        content = DocBody(html) 
        URLList, URLs = DocURLList(html, PageURL)
    except Exception:
        print("Wikipedia article not valid. Skipping.")
        #UniqueURLs.remove(i)
        Peripheral -= 1
        flag = True
        pass

    if flag == False:
        #Write the data to a new/the respective text file
        filename = Directory + str(Title) + ".txt"
        file = open(filename, 'w')
        file.write(PageURL + '\n')
        file.write(Title + '\n')
        file.write(str(Date) + '\n')
        file.write(content + '\n')
        file.close()

        #This file contains the file names in order
        Namefile = open((Directory + "Names.txt"), 'a+')
        Namefile.write(Title +'\n')
        Namefile.close()
    
        #This contains the title followed by its respective URLs
        URLFile = open((Directory + "IndividualURLList.txt"), 'a+')
        URLFile.write(PageURL + ' ' + URLs + '\n')
        URLFile.write('')
        URLFile.close()
        count += 1
    print("Completed scraping peripheral file ", count, " out of ", Peripheral, ": ", Title)
    time.sleep(1.5)

#Minor corrections
TrimPeripheralURLs(CorePages + Peripheral)

metadata = open(Directory + 'metadata.txt', 'w')
metadata.write(str(CorePages) + '\n')
metadata.write(str(Peripheral))
metadata.close()

print("Webcrawl completed")