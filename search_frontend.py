#========= IMPORTS =======

import sys
from collections import Counter, OrderedDict
import itertools
import math
from itertools import islice, count, groupby
from inverted_index_gcp import *
import nltk as nltk
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing
from flask import Flask, request, jsonify
from collections import defaultdict
from contextlib import closing
import hashlib
from pathlib import Path
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords')
import gzip
import csv

# =========== start HELPER FUNCTIONS TO READ PKL FILES ======

def write_disk(base_dir, name, my_object):
    with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
        pickle.dump(my_object, f)


def read_from_disk(base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


# ===============TOKENIZER OF GCP PART HOMEWORK ======

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124

def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS


def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower())]


def filter_tokens(tokens):
    ''' The function takes a list of tokens, filters out `all_stopwords`'''
    filtered_tokens = []
    for word in tokens:
        if (word in all_stopwords):
            continue
        else:
            filtered_tokens.append(word)

    return filtered_tokens

def word_count(text, id):
  ''' Count the frequency of each word in `text` (tf) that is not included in
  `all_stopwords` and return entries that will go into our posting lists.
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs
      for example: [("Anarchism", (12, 5)), ...]
  '''
  tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
  tokens = filter_tokens(tokens)
  # YOUR CODE HERE
  word_counts_counter = Counter(tokens)
  word_counts_list = word_counts_counter.items()
  list_to_return = []
  for item in word_counts_list:
    list_to_return.append((item[0], (id, item[1])))
  #raise NotImplementedError()
  return list_to_return


def FirstElement(elem):
    return elem[0]


def reduce_word_counts(unsorted_pl):
    ''' Returns a sorted posting list by wiki_id.
    Parameters:
    -----------
      unsorted_pl: list of tuples
        A list of (wiki_id, tf) tuples
    Returns:
    --------
      list of tuples
        A sorted posting list.
    '''
    # YOUR CODE HERE
    sorted_pl = sorted(unsorted_pl, key=FirstElement)
    # raise NotImplementedError()
    return sorted_pl


def calculate_df(postings):
    ''' Takes a posting list RDD and calculate the df for each token.
    Parameters:
    -----------
      postings: RDD
        An RDD where each element is a (token, posting_list) pair.
    Returns:
    --------
      RDD
        An RDD where each element is a (token, df) pair.
    '''
    # YOUR CODE HERE
    # raise NotImplementedError()
    return postings.map(lambda x: (x[0], len(x[1])))


NUM_BUCKETS = 124


def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS



def sum_list(my_list):
    my_sum = 0
    for num in my_list:
        my_sum += num
    return my_sum


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
from contextlib import closing


def read_posting_list(inverted, w):
    base_dir = "postings_gcp/"
    with closing(MultiFileReader()) as reader:
        locs = [(base_dir + inverted.posting_locs[w][0][0], inverted.posting_locs[w][0][1])]

        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)

        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')

            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list

def cosine_similarity(index,query):
  """
  index : inverted index loaded from the corresponding files.
  query : a string

  this function calc cosine similarity score for all the relevant docs of the query

  return: dictionary with key = doc_id, val = cosine similarity score
  """

  query_tokens = filter_tokens(tokenize(query))
  len_query = len(query_tokens)

  doc_id_score_dict = {}
  query_counter = Counter(query_tokens)
  query_norma=0
  for term, tf in query_counter.items():
    query_norma += tf**2
  query_norma = math.sqrt(query_norma)

  for term in query_tokens:
    try:
      pl = read_posting_list(index, term)
    except:
      continue
    idf = index.idf.get(term)
    for doc_id, tf in pl:
        if doc_id==0 or tf==0:
            continue
        doc_norma = index.doc_norma.get(doc_id)
        if index.doc_length.get(doc_id) == None:
            continue

        cosine_similarity = ((query_counter.get(term)/len_query)*(tf/index.doc_length.get(doc_id)) * idf)/(query_norma*doc_norma)
        doc_id_score_dict[doc_id] = doc_id_score_dict.get(doc_id,0) + cosine_similarity


  return doc_id_score_dict


def get_top_n(sim_dict, N=100):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    if len(sim_dict) == 0:
        return []

    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


def similarity_title(index, query):
    """
    index : inverted index loaded from the corresponding files.
    query : a string

    this function calc cosine similarity score for all the relevant docs of the query

    return: dictionary with key = doc_id, val = cosine similarity score
    """
    query_tokens = filter_tokens(tokenize(query))
    doc_id_score_dict = {}

    for term in query_tokens:
        try:
            pl = read_posting_list(index, term)
        except:
            continue
        for doc_id, tf in pl:
            if doc_id == 0 or tf==0:
                continue
            doc_id_score_dict[doc_id] = doc_id_score_dict.get(doc_id, 0) + tf

    return doc_id_score_dict

def intersection(l1,l2):
    """
    This function perform an intersection between two lists.

    Parameters
    ----------
    l1: list of documents. Each element is a doc_id.
    l2: list of documents. Each element is a doc_id.

    Returns:
    ----------
    list with the intersection (without duplicates) of l1 and l2
    """
    return list(set(l1)&set(l2))


def precision_at_k(true_list, predicted_list, k=40):
    """
    This function calculate the precision@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    predicted_list_k = predicted_list[:k]
    mone = intersection(predicted_list_k, true_list)
    mahane = predicted_list[:k]
    return float("{:.3f}".format(len(mone) / len(mahane)))


def average_precision(true_list, predicted_list, k=40):
    """
    This function calculate the average_precision@k metric.(i.e., precision in every recall point).

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, average precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    predicted_list_k = predicted_list[:k]
    list_of_precision = []

    for doc in predicted_list_k:
        if doc in true_list:
            precision_score = precision_at_k(true_list, predicted_list_k, predicted_list_k.index(doc) + 1)
            list_of_precision.append(precision_score)

    if len(list_of_precision) == 0:
        return 0.0

    return sum_list(list_of_precision) / len(list_of_precision)



# =========== end HELPER FUNCTIONS TO READ PKL FILES ======


# ======READ 3 inverted indecies FROM DISC======:
base_dir = "postings_gcp/"
index_text_readed = read_from_disk(base_dir, "index_text")
index_title_readed = read_from_disk(base_dir, "index_title")
index_anchor_readed = read_from_disk(base_dir, "index_anchor")

# ======== READ PAGE RANK =============
pr_file_dir = "pr/pr.csv.gz"
f = gzip.open(pr_file_dir, mode='rt')
csvobj = csv.reader(f, delimiter=',', quotechar="'")
page_rank_dict = {int(rows[0]): float(rows[1]) for rows in csvobj}

# =========== READ PAGE VIEW ==========

pv_dir = "pv/"
pv_name = "pageviews-202108-user"
page_view_dict = read_from_disk(pv_dir, pv_name)

# ============ READ DOC2TITLE =========
pv_dir = "postings_gcp/"
pv_name = "doc2title"
doc2title_dict = read_from_disk(pv_dir, pv_name)

#============= MAX VALUES OF PR AND PV ===============

max_pr = sorted(page_rank_dict.values(), reverse=True)[0]
max_pv = sorted(page_view_dict.values(), reverse=True)[0]


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

def convert_to_final(tuple_list):
  if len(tuple_list)==0:
      return []
  finell_list =[]

  for item in tuple_list:
    try:
        doc_name = doc2title_dict.get(item[0])
    except:
        continue

    finell_list.append((item[0], doc_name))

  return finell_list



def MySearch(query, w_titlw, w_body, w_anchor=None, w_page_rank=None, w_pageviews=None):
    """
    a function that calculate weighted avarage of all parameters to one single score
    and returns a dictionary  with weighted average score per doc id.
    :param query: str of a string -> we will do tokenization to it in the function
    :param w_titlw: score to give the similarity between the query and wiki titles
    :param w_body: score to give the similarity between the query and wiki text pages
    :param w_anchor:  score to give the similarity between the query and wiki anchors
    :param w_page_rank: score to the page rank of the  document that have similarity to our query
    :param w_pageviews: score to the page views of the  document that have similarity to our query
    :return: a dictionary with weighted average score while {(doc_id,final_score)...}
    """

    # body

    dict_score_cosin = cosine_similarity(index_text_readed, query)

    topN_cosin = get_top_n(dict_score_cosin, 100)
    if len(topN_cosin)!= 0:
        max_val_cosin = topN_cosin[0][1]
        topN_cosin = [(item[0], item[1] / max_val_cosin) for item in topN_cosin]
    else:
        topN_cosin = []
    # title
    dict_score_title = similarity_title(index_title_readed, query)
    topN_title = get_top_n(dict_score_title, 100)
    if len(topN_title) != 0:
        max_val_title = topN_title[0][1]
        topN_title = [(item[0], item[1] / max_val_title) for item in topN_title]
    else:
        topN_title = []

    # anchor
    if w_anchor != None:
        dict_score_anchor = similarity_title(index_anchor_readed, query)
        topN_anchor = get_top_n(dict_score_anchor, 100)
        if len(topN_anchor) != 0:
            max_val_anchor = topN_anchor[0][1]
            topN_anchor = [(item[0], item[1] / max_val_anchor) for item in topN_anchor]
        else:
            topN_anchor = []

    dic_final_score = {}
    for doc, score in topN_cosin:
        dic_final_score[doc] = dic_final_score.get(doc, 0) + score * w_body

    for doc, score in topN_title:
        dic_final_score[doc] = dic_final_score.get(doc, 0) + score * w_titlw

    if w_anchor != None:
        for doc, score in topN_anchor:
            dic_final_score[doc] = dic_final_score.get(doc, 0) + score * w_anchor



    if w_page_rank != None and w_pageviews != None:
        for doc, score in dic_final_score.items():
            dic_final_score[doc] = dic_final_score.get(doc) + (w_page_rank * (page_rank_dict.get(doc, 0) / max_pr)) + (
                        w_pageviews * (page_view_dict.get(doc, 0) / max_pv))

    if w_page_rank != None and w_pageviews == None:
        for doc, score in dic_final_score.items():
            dic_final_score[doc] = dic_final_score.get(doc) + (w_page_rank * (page_rank_dict.get(doc, 0) / max_pr))

    if w_page_rank == None and w_pageviews != None:
        for doc, score in dic_final_score.items():
            dic_final_score[doc] = dic_final_score.get(doc) + (w_pageviews * (page_view_dict.get(doc, 0) / max_pv))

    return dic_final_score


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)

    # BEGIN SOLUTION
    #calc final weighted average score per each doc id
    final_dict = MySearch(query, 0.2,0.4, 0.2, 0.1, 0.1)
    top100 = get_top_n(final_dict)
    # adding titles and delleting scores
    res = convert_to_final(top100)
    # END SOLUTION

    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)

    # BEGIN SOLUTION
    #calc tfidf per each doc id in the text inverted index
    final_dict = cosine_similarity(index_text_readed, query)
    top100 = get_top_n(final_dict)
    # adding titles and deleting scores
    res = convert_to_final(top100)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    #calc term frquency in the title index how many times the query term appers in the title
    final_dict = similarity_title(index_title_readed, query)


    sorted_scores_tuples_list = sorted(final_dict.items(),key =  lambda x: x[1],reverse = True)

    #adding titles and delleting scores
    res = convert_to_final(sorted_scores_tuples_list)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)

    # BEGIN SOLUTION
    # calc term frquency in the anchor index how many times the query term appers in the anchor doc

    final_dict = similarity_title(index_anchor_readed, query)
    #new
    if len(final_dict)==0:
        return jsonify(res)

    sorted_scores_tuples_list = sorted(final_dict.items(), key=lambda x: x[1], reverse=True)
    # adding titles and deleting scores
    res = convert_to_final(sorted_scores_tuples_list)
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for docid in wiki_ids:
        #returns the page rank per docid if not exists-> pagerank =0
        res.append(page_rank_dict.get(docid,0))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for docid in wiki_ids:
        #returns the page views number per docid if not exists-> pagerank =0
        res.append(page_view_dict.get(docid,0))
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)

# Let's start with a small block size of 30 bytes just to test things out.
BLOCK_SIZE = 1999998




