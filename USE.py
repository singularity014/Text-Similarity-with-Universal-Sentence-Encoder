import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os, sys
from sklearn.metrics.pairwise import cosine_similarity
# ----------------------------------------------------
with open('test_kwds', 'r') as fil:
    sentences_list = fil.readlines()
sentences_list = list(map(lambda x:x.strip('\n'), sentences_list))

# ---------------------------------------------------
def cos_sim(input_vectors):
    similarity = cosine_similarity(input_vectors)
    return similarity

# get topN similar sentences
def get_top_similar(sentence, sentence_list, similarity_matrix, topN):
    # find the index of sentence in list
    index = sentence_list.index(sentence)
    # get the corresponding row in similarity matrix
    similarity_row = np.array(similarity_matrix[index, :])
    # get the indices of top similar
    indices = similarity_row.argsort()[-topN:][::-1]
    return [sentence_list[i] for i in indices]
# ----------------------------------------------------------------
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
# module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)
# ---------------------------------------------------------------
# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  sentences_embeddings = session.run(embed(sentences_list))

similarity_matrix = cos_sim(np.array(sentences_embeddings))
# sentence = "family weekend and travel"

thematic = "family weekend"
top_similar = get_top_similar(thematic, sentences_list, similarity_matrix, 15)


print('Thematic/Query - {} \n'.format(thematic))
print('----------------------------------------')

# printing the list using loop
for x in range(len(top_similar)):
    print(top_similar[x])

# ----------------------------------------------
