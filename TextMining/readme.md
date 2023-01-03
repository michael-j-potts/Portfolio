## Project overview - Wikipedia Webcrawler with the page rank algorithm
[Video link](https://drive.google.com/file/d/1B-8o4xKV_p3TTEfY1extllZnv511Jy95/view?usp=sharing)

Out of respect for wikipedia's server load, please keep web crawl low and choose only a few core articles to mine (2-4). 2-4 will gather typically 200-2000 articles to make a subgraph out of. Core articles with a large amount of peripheral URLs will have a higher pagerank.

### Wikipedia Webcrawler
To begin, the wikipedia webcrawler will take user input to ask how many core documents they would like. The core documents help to create a subgraph to run the page rank algorithm on. The user may choose to search for a specific topic, or settle for a random topic for the first document only. This implementation was chosen over a download of the wikimedia dumps available, as the complexity of the pagerank algorithm increases per document, which would result in extraordinary runtimes. Each core document is crawled to gather its contents, from which each wikipedia url within the article body is collected and stored. To ensure relation of documents, the remaining core documents are chosen at random from the list of URLs gathered from the first document. Once the core documents have been crawled, each of the remainig wikipedia URLs (refered to as peripheral URLs/peripheral documents) are gathered as well to create a graph network. As another step to ensure only graph connections that connect to known nodes (documents) are retained, each peripheral document URL list was refined to only keep URLs in the total document list. 
The files/folders created by the webcrawler are as follows:
folder - "repository" - This folder contains all created folders and files from the webcrawler
file - "IndividualURLList.txt" - a file which contains the wikipedia URLs for each documents article
file - "Names.txt" - The document titles in order that each article was scraped
file - "CoreURLs.txt" - The URLs generated from only the core documents. This was used as an internal program file to generate the list of peripheral documents
file - "metadata.txt" - this is an internal file passed to the page rank algorithm that simply records the number of core and peripheral files scraped in two lines (# of core files, # of peripheral files.)
file - "Trimmed.txt" - This is the result of the peripheral file trimming process to ensure the graph structure only contains nodes that point to other known nodes. It contains the list of each documents URLs which point to other documents in the given list. 
file - "{title}.txt" - Each document articles contents was scraped (including title, url, date, and contents) into the respectively named document title text file. This was to be used for PLSA implementation but I ran out of time to implement PLSA.

### Page Rank Algorithm
The next program is the page rank alogirthm and some preprocessing scripts. This program imports Trimmed.txt, Names.txt, and metadata.txt. It also sets a variety of elements according to the page rank algorithm and the content covered in week 5 of Text Information systems at UIUC. The total number of n documents is gathered from metadata.txt, which is then combine with the Trimmed.txt file which creates a graph edge adjacency matrix. This edge matrix is then used to create a transition matrix which is used by the page rank algorithm. The transition matrix calculate the total number of URLs present in each document and represents the mask as 1/URLpresent (where URLpresent is the number of collection URLs that are true in each document). The page rank function then takes the transition matrix, a pre-determined alpha score, the total number of documents in the collection (n), a predetermined maximum iteration score and a predetermined epsilon score to calculate the page rank score for each document. Once either the sum of the absolute difference of the previous and current P vector values fall under epsilon, or the number of max iterations is reached, the program returns the Final Pvector score which contains the pagerank for each document. 
The program finishes by displaying the 5 highest and the 5 lowest page rank scores.

## Team Member Contributions
I was the only contributor.

## Related work and Used Libraries/Models/Previous Projects
The code in this project (aside from libraries, theory and mathematical formuli) is original. The primary libraries I relied upon are beautiful soup, the wikipedia library (for the search query only, not HTML/XML processing), requests (to process websearches), datetime, re, time, os, and numpy. My primary source of material for learning and implementation of the mathematics involved included the week 5 material from Text information systems from UIUC, the textbook "Text Data Management and Analysis" chapter 10.3.1 "PageRank", wikipedia's article on pagerank, and the three websites: https://www.geeksforgeeks.org/page-rank-algorithm-implementation/, https://towardsdatascience.com/pagerank-algorithm-fully-explained-dc794184b4af, and http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html.
From these sources, I found the UIUC material and the text book especially valuable.

## Code structure
#### Webcralwer.py
input("How many core documents would you like to collect? (int)")

input("Do you have a topic you would like to start with? (y/n))

*Optional* input("What is your query search")

*Will gather the number of core and subsequent peripheral documents automatically*

*for n in number of core articles, the title, pageurl, Date, and content will be gathered and written to {title}.txt. Further the list of Core and peripheral URLs will be gathered and written to CoreURLs.txt and IndividualURLList.txt respectively. The list of names is also gathered in the order they were collected. The remaining CoreURLs are chosen at random from the first articles scraped URL lis to ensure relevance.* 

*The CoreURLs are removed from the list of remaining URLs to gather.*

*for i in remaining URLs, the program continues in the same manor as the Core list.*

*TrimPeripheralURLs is called, which removes all URLs from all articles that dont directly relate to one of the gathered articles (each file creates a sub graph that only considers the files that have been gathered). Further, this is written to a file Trimmed.txt, which will be used for the edge connection graph in PageRank.py.*

*A metadata file containing the number of core and peripheral articles is written to a file metadata.txt to be used in PageRank.py*

*Webcrawl completed*
#### PageRank.py
*Several global elements are set including graphconnections which is Trimmed.txt (the file containing each n articles URLs written on n lines), alpha, n (the number of articles gathered), max iterations, and epsilon.*

*TransitionMatrix is called, which calls GraphMatrix.*

*GraphMatrix creates a binary mask matrix for if a given article has 1 of the n articles gathered in its URL list, it will be 1, Else 0.*

*This mask is returned to the TransitionMatrix, which will count the trues per article and divide the row of 1s by the number of trues per row.*

*Next PageRank is called which calculates the variable a and the first initiallization of the Pvector (as 1/# of articles). The program then enters a iteration loop which matrix multiplies a to the first initialized Pvector. the sum of the absolute subraction of the current Pvector by tthe Initial Pvector is compareed against epsilon. the current Pvector is set as the previous, and the loop repeats until convergence or max iterations have been achieved.*

*The Pvector of each article is returned, and sorted to discover the 5 highest and 5 lower pageranked articles.*

## Detailed instructions for reviewers to set up and run code, including possible errors or blockers
To run the program, execute the file run.exe, then follow the prompts. First begin by making each .sh file executable:

`chmod u+x partialrun.sh`

`chmod u+x fullrun.sh`

To run the program with the pregathered repository:

`bash partialrun.sh`

To run the full program and manually gather the repository:

`bash fullrun.sh`

Some possible errors that might be encountered include 1).a connect error, 2).timeout error, 3).a core file failed scrape or possibly 4).a naming error.

To resolve each error, simply rerun the program. 

To counteract these errors, I have implemented:
1). This may be caused by port exhaustion. It may be affected by heavier internet usage. Some things to try to assist are: A request speed of 1 article every 1.5 seconds, and a small total gather rate of 2-4 core articles (200-2000 average articles).
2). A try and exception clause for the request.get() functions which include a second try on the same URL after a 5 second timeout.
3). If a core file fails to scrape (due to the article not existing or being heavily incomplete), the count will become inaccurate and cause the PageRank program to fail. Restart the program to ensure all core files scrape successfully.
4). Article Title monitoring through replacing / and . in a title with - and nothing respectively. This prevents file save errors when attempting to save the file as the titles name (ex .300_Winchester_Magnum) but preserves the URL as necessary.
