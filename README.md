# Wikipedia Search Engine
IR Project in BGU university<br/><br/>
<picture>
  <img alt="Wikipedia Curpos" src="https://upload.wikimedia.org/wikipedia/en/thumb/8/80/Wikipedia-logo-v2.svg/1200px-Wikipedia-logo-v2.svg.png" width="250" height="250">
</picture><br/>
## Index Building <br/>
[indexescreation.py]

In order to retrieve the information quickly and efficiently, we have set up ahead of time three different inverted index objects based on the object provided to us in Task 3 which maintain in their fields useful calculation results aimed at shortening the retrieval time<br/>
+ **Index_text_readed** -inverted index that includes all the data of the body text of Wikipedia pages. In order to facilitate the calculations of tf-idf and cosine similarity the object also maintains the following fields:<br/>

   Self.N - The amount of all documents in the current inverted index.<br/>

  Self.idf - A dictionary that maps each term to the idf value corresponding to the term.<br/>

  Self.doc_length - A dictionary that mapseach doc_id to its length<br/>

  Self.norma - A dictionary that maps each doc_id to its norma value calculated by the sum of the tf-idf values of all the words in the document, and finally a squart on the sum.(this value used in calculating the cosine similarity by tf-idf)<br/>
+ **Index_title _readed** -inverted index that integrates all the data of the titles of the Wikipedia pages and maintains the original fields of the object.<br/>
+ **Index_anchor _readed** -inverted index that integrates all the data of pages that appeared as links in Wikipedia pages and maintains the original fields of the object.<br/>
+ **Page_rank_dict** -A dictionary that maps each page in Wikipedia to its page rank value<br/>
+ **Page_views_dict** - A dictionary that maps each page of Wikipedia to its pageviews value.<br/>
+ **Max_pv** - A variable that maintains the maximum Pageviews value of all Wikipedia pages for normalization purposes.<br/>
+ **Max_pr** - A variable that maintains the maximum pagerank value of all Wikipedia pages for normalization purposes.<br/>

we uploaded all the indecies files we created to the VM instance before loading search_frontend.py
the file that calculate the indecies is by using spark methods and gcp.<br/>


## Searching Engine Functionality<br/>
[search_fronted.py]<br/><br/>
**global variables** - firat of all, we read to global variables all the data structures we prepared above. <br/>
**Search_body()** - The function retrieves the 100 documents with the highest similarity value based on cosine similarity.<br/>
**Search_title()** - The function retrieves all the documents in the title that contained the query words. Using the Posting list, for each document the number of times the. different query words appear in it is summarized, and finally the documents are displayed in descending order according to this value.<br/>
**Search_anchor()** - The function retrieves all documents linked from other pages that contained the query words in anchor_text. Using the Posting list, for each document the number of times the different query words appear in it is summarized, and finally the documents are displayed in descending order according to this value.<br/>
**Get_pagerank()** - The function retrieves the Pagerank values for the docid values it received.<br/>
**Get_pagerank()** - The function retrieves the pageviews values for the docid values it received.<br/>
**Search()** - The main search function we chose for our engine is aided by the values returned from the basic functions above.<br/>
the function performs a weighted average on the values returned from the functions mentioned above as follows:<br/>
- Retrieve the top100 most similar documents returned from search_body and normalize the scores at the maximum value obtained for the highest doc_id.<br/>
- Retrieve the top100 most similar documents returned from search_title and normalize the scores with the maximum value obtained for the highest doc_id<br/>
- Retrieve the top100 most similar documents returned from search_anchor and normalize the scores at the maximum value obtained for the highest doc_id<br/>
- For all the documents we received from the three retrievals above - calculate the Page_rank values for each document and normalize the overall highest value obtained for the page rank scores of all Wikipedia pages.<br/>
- For all the documents we received from the three retrievals above - calculate the Page_view values for each document and normalize to the highest overall value obtained for the page view scores of all Wikipedia pages.<br/>
- each component in the score receiving a specific weight (all weights add up to 1).<br/>
- We initialized a dictionary at which each doc_id mapped to the summation of its weighted normalized scores from each of the steps. <br/>
- finally we return the documents according to the weighted score they received in descending order.<br/>

