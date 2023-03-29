import re
import math
import random
import numpy as np
from numpy.linalg import norm
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple
from nltk.stem.snowball import SnowballStemmer


### File IO and processing
def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])


stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')


class Document(NamedTuple):
    doc_id: int
    text: List[str]

    def sections(self):
        return [self.text]

    def __repr__(self):
        return f"doc_id: {self.doc_id}\n" + f"  text: {self.text}"


def read_docs(file, model):
    docs = []
    category = []

    with open(file) as f:
        for line in f:
            line = line.strip().split('\t')
            if model == 'bow':
                # bag of words
                category.append(int(line[1]))
                text = [word for word in line[2].strip().split()]
                docs.append(text)
            elif model == 'bigram':
                # bigrams
                sentence = line[2]
                text = [word for word in line[2].strip().split()]
                pair = []
                category.append(int(line[1]))
                t = re.findall(r'\.X-\S+', sentence)
                if len(t) > 0:
                    t = t[0]
                t_index = text.index(t)
                pair.append(text[t_index - 1] + ' ' + text[t_index])
                pair.append(text[t_index] + ' ' + text[t_index + 1])
                docs.append(pair)

    return [Document(i + 1, text) for i, text in enumerate(docs)], category


def profile(vecs: Dict[str, int], target: int, category: list):
    profile = defaultdict(float)
    count = 0
    t_vec = []
    for vec in vecs:
        if category[count] == target:
            t_vec.append(vec)
        count += 1

    for vec in t_vec:
        for key in vec.keys():
            profile[key] += vec[key]

    N = len(t_vec)
    for key in profile.keys():
        profile[key] /= N

    return dict(profile)


def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
                                  for sec in doc.sections()])


def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]


def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stopwords]
                                  for sec in doc.sections()])


def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]


def compute_doc_freqs(docs: List[Document]):
    freq = Counter()
    for doc in docs:
        for text in doc.text:
            freq[text] += 1
    return freq


def compute_tf(doc: Document, weights: list):
    vec = defaultdict(float)
    pos = 0
    for word in doc.text:
        vec[word] += weights[pos]
        pos += 1

    return dict(vec)


def compute_tfidf(doc: Document, doc_freqs: Dict[str, int], weights: list, doc_len: int):
    vec = defaultdict(float)
    term_freq = compute_tf(doc, weights)
    for term in term_freq.keys():
        tf = term_freq[term]
        if term in doc_freqs.keys():
            df = doc_freqs[term]
            vec[term] = tf * np.log(doc_len / df)
        else:
            vec[term] = 0
    return dict(vec)


### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    """
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    """
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)


def cosine_sim(x, y):
    """
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    """
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))


def sum_profiles(vecs: Dict[str, int], target: int, category: list):
    profile = defaultdict(float)
    count = 0
    t_vec = []
    for vec in vecs:
        if category[count] == target:
            t_vec.append(vec)
        count += 1

    for vec in t_vec:
        for key in vec.keys():
            profile[key] += vec[key]

    return dict(profile)


def exp_weight(docs, mode):
    w_lst = []
    if mode == 'bow':
        for doc in docs:
            doc_w = []
            target = None
            for text in doc.text:
                target = re.findall(r'\.X-\S+|\.x-\S+', text)
                if len(target) > 0:
                    target = target[0]
                    break
            for word in doc.text:
                if word == target:
                    doc_w.append(0)
                else:
                    doc_w.append(1 / abs(doc.text.index(target) - doc.text.index(word)))
            w_lst.append(doc_w)
    elif mode == 'bigram':
        for doc in docs:
            doc_w = [1, 1]
            w_lst.append(doc_w)

    return w_lst


def uniform_weight(docs, mode):
    w_lst = []
    for doc in docs:
        tmp = []
        for word in doc.text:
            tmp.append(1)
        w_lst.append(tmp)
    return w_lst


def stepped_weight(docs, mode):
    w_lst = []
    if mode == 'bow':
        for doc in docs:
            tmp = []
            target = None
            for text in doc.text:
                target = re.findall(r'\.X-\S+|\.x-\S+', text)
                if len(target) > 0:
                    target = target[0]
                    break
            for word in range(len(doc.text)):
                if abs(doc.text.index(target) - word) > 3:
                    tmp.append(1)
                elif abs(doc.text.index(target) - word) == 2 or abs(doc.text.index(target) - word) == 3:
                    tmp.append(3)
                elif abs(doc.text.index(target) - word) == 1:
                    tmp.append(6)
                else:
                    tmp.append(0)
            w_lst.append(tmp)

    elif mode == 'bigram':
        for doc in docs:
            tmp = [6, 6]
            w_lst.append(tmp)
    return w_lst


def custom_weight(docs, mode):
    """
    Implements the exponential decay with window threshold of 5 (context size of 5 on left and right)
    Works well to restrict the context size. If we don't restrict the context size, the words that are far away will
    affect the weight of the current term, no matter however small it might be, even if it's not relevant to the current
    word. Hence, it helps to have some kind of a context window, outside which the weight of all terms will be 0.
    """
    w_lst = []
    if mode == 'bow':
        for doc in docs:
            tmp = []
            target = None
            for text in doc.text:
                target = re.findall(r'\.X-\S+|\.x-\S+', text)
                if len(target) > 0:
                    target = target[0]
                    break
            for word in range(len(doc.text)):
                if abs(doc.text.index(target) - word) > 4:
                    tmp.append(0)
                elif abs(doc.text.index(target) - word) == 4:
                    tmp.append(10)
                elif abs(doc.text.index(target) - word) == 3:
                    tmp.append(100)
                elif abs(doc.text.index(target) - word) == 2:
                    tmp.append(1000)
                elif abs(doc.text.index(target) - word) == 1:
                    tmp.append(10000)
                elif abs(doc.text.index(target) - word) == 0:
                    tmp.append(0)
            w_lst.append(tmp)

    elif mode == 'bigram':
        for doc in docs:
            tmp = [10, 10]
            w_lst.append(tmp)
    return w_lst


def bayes_weight(V_sum1, V_sum2, epsilon):
    Llike = dict()
    for term in V_sum2.keys():
        if V_sum1.get(term) is None or V_sum2[term] == 0:
            continue
        if V_sum1[term] > 0:
            Llike[term] = math.log(V_sum1[term] / V_sum2[term])
        else:
            Llike[term] = math.log(epsilon / V_sum2[term])
    for term in V_sum1.keys():
        if term not in V_sum2 and V_sum1[term] != 0:
            Llike[term] = math.log(V_sum1[term] / epsilon)
    return Llike


def smsspam():
    docs, category = read_docs('smsspam-train.tsv', 'bow')
    # compute doc frequency
    doc_freq = compute_doc_freqs(docs)
    # compute weights
    weight = uniform_weight(docs, 'bow')
    # compute tfidf
    doc_count = 0
    tf = []
    N = len(docs)
    for doc in docs:
        tf.append(compute_tfidf(doc, doc_freq, weight[doc_count], N))
        doc_count += 1
    profile1 = profile(tf, 1, category)
    profile2 = profile(tf, 2, category)

    dev1_docs, categories = read_docs('smsspam-dev.tsv', 'bow')
    # compute doc frequency
    doc_freq = compute_doc_freqs(dev1_docs)
    # compute weights
    weight = uniform_weight(dev1_docs, 'bow')
    # compute tfidf
    doc_count = 0
    tf = []
    N = len(dev1_docs)
    for doc in dev1_docs:
        tf.append(compute_tfidf(doc, doc_freq, weight[doc_count], N))
        doc_count += 1
    sim1 = []
    sim2 = []
    for vec in tf:
        sim1.append(cosine_sim(vec, profile1))
        sim2.append(cosine_sim(vec, profile2))
    right = 0
    wrong = 0
    for i in range(len(sim1)):
        if sim1[i] > sim2[i] and categories[i] == 1:
            right += 1
        elif sim1[i] > sim2[i] and categories[i] == 2:
            wrong += 1
        elif sim2[i] > sim1[i] and categories[i] == 2:
            right += 1
        elif sim2[i] > sim1[i] and categories[i] == 1:
            wrong += 1
        elif sim1[i] == sim2[i]:
            right += 1
    acc_sms = right / (right + wrong)
    return acc_sms


def experiment():
    train_docs = ['tank-train.tsv', 'perplace-train.tsv', 'plant-train.tsv', 'smsspam-train.tsv']
    dev_docs = ['tank-dev.tsv', 'perplace-dev.tsv', 'plant-dev.tsv', 'smsspam-dev.tsv']
    stem_funcs = {
        'stemmed': True,
        'unstemmed': False
    }
    weight_funcs = {
        '#0-uniform': uniform_weight,
        '#1-expndecay': exp_weight,
        '#2-stepped': stepped_weight,
        '#3-yours': custom_weight
    }
    model_funcs = {
        '#1-bag-of-words': 'bow',
        '#2-adjacent-separate-LR': 'bigram'
    }
    comb = [
        ['unstemmed', '#0-uniform', '#1-bag-of-words'],
        ['stemmed', '#1-expndecay', '#1-bag-of-words'],
        ['unstemmed', '#1-expndecay', '#1-bag-of-words'],
        ['unstemmed', '#1-expndecay', '#2-adjacent-separate-LR'],
        ['unstemmed', '#2-stepped', '#1-bag-of-words'],
        ['unstemmed', '#3-yours', '#1-bag-of-words']
    ]

    print('Stemming', 'Position Weighting', 'Local Collocation Modelling', 'tank', 'pers/place', 'plants', 'smsspam',
          sep='\t')

    acc_sms = round(smsspam(), 4)

    for stemming, weighting, model in comb:
        accuracy = []
        for doc_i in range(0, len(train_docs)):
            if doc_i == len(train_docs) - 1:
                # handle smsspam files since they don't have -X. tokens
                accuracy.append(acc_sms)
                acc_sms = "--"
                break
            docs, category = read_docs(train_docs[doc_i], model=model_funcs[model])
            # check for stem doc
            if stemming == 'stemmed':
                docs = stem_docs(docs)
            # compute doc frequency
            doc_freq = compute_doc_freqs(docs)
            # compute weights
            weight = weight_funcs[weighting](docs, model_funcs[model])
            # compute tfidf
            doc_count = 0
            tf = []
            N = len(docs)
            for doc in docs:
                tf.append(compute_tfidf(doc, doc_freq, weight[doc_count], N))
                doc_count += 1
            profile1 = profile(tf, 1, category)
            profile2 = profile(tf, 2, category)

            dev1_docs, categories = read_docs(dev_docs[doc_i], model=model_funcs[model])
            # check for stem doc
            if stemming == 'stemmed':
                dev1_docs = stem_docs(dev1_docs)
            # compute doc frequency
            doc_freq = compute_doc_freqs(dev1_docs)
            # compute weights
            weight = weight_funcs[weighting](dev1_docs, model_funcs[model])
            # compute tfidf
            doc_count = 0
            tf = []
            N = len(dev1_docs)
            for doc in dev1_docs:
                tf.append(compute_tfidf(doc, doc_freq, weight[doc_count], N))
                doc_count += 1
            sim1 = []
            sim2 = []
            for vec in tf:
                sim1.append(cosine_sim(vec, profile1))
                sim2.append(cosine_sim(vec, profile2))
            right = 0
            wrong = 0
            for i in range(len(sim1)):
                if sim1[i] > sim2[i] and categories[i] == 1:
                    right += 1
                elif sim1[i] > sim2[i] and categories[i] == 2:
                    wrong += 1
                elif sim2[i] > sim1[i] and categories[i] == 2:
                    right += 1
                elif sim2[i] > sim1[i] and categories[i] == 1:
                    wrong += 1
                elif sim1[i] == sim2[i]:
                    right += 1
            acc = right / (right + wrong)
            accuracy.append(acc)
        print(stemming, weighting, model, *accuracy, sep='\t')


def bayes():
    train_docs = ['tank-train.tsv', 'perplace-train.tsv', 'plant-train.tsv', 'smsspam-train.tsv']
    dev_docs = ['tank-dev.tsv', 'perplace-dev.tsv', 'plant-dev.tsv', 'smsspam-dev.tsv']
    print('stemmed', '#1-expndecay', '#1-bag-of-words', 'bayes', end='\t')
    for doc_i, (train, dev) in enumerate(zip(train_docs, dev_docs)):
        if doc_i == len(train_docs) - 1:
            # handle smsspam files since they don't have -X. tokens
            # accuracy.append(acc_sms)
            # acc_sms = "--"
            print('--')
            break

        docs, category = read_docs(train, model='bow')

        # check for stem doc
        docs = stem_docs(docs)

        # remove stopwords
        docs = remove_stopwords(docs)

        # compute weights
        weight = exp_weight(docs, 'bow')

        # compute tf
        doc_count = 0
        tf = []
        for doc in docs:
            tf.append(compute_tf(doc, weight[doc_count]))
            doc_count += 1
        vsum1 = sum_profiles(tf, 1, category)
        vsum2 = sum_profiles(tf, 2, category)
        LLike = bayes_weight(vsum1, vsum2, 0.2)

        docs, category = read_docs(dev, model='bow')
        # check for stem doc
        docs = stem_docs(docs)
        # remove stopwords
        docs = remove_stopwords(docs)
        # compute weights
        weight = exp_weight(docs, 'bow')
        # compute tf
        doc_count = 0
        tf = []
        for doc in docs:
            tf.append(compute_tf(doc, weight[doc_count]))
            doc_count += 1

        preds = []
        right = 0
        wrong = 0
        for i, vtest in enumerate(tf):
            sumofLL = 0
            for term in vtest.keys():
                sumofLL += LLike.get(term, 0) * vtest[term]
            if sumofLL > 0:
                pred = 1
                preds.append(1)
            elif sumofLL < 0:
                pred = 2
                preds.append(2)
            else:
                pred = random.choice([1, 2])
                preds.append(pred)
            if pred == category[i]:
                right += 1
            else:
                wrong += 1
        print((right / (right + wrong)), end='\t')


if __name__ == '__main__':
    experiment()
    bayes()
