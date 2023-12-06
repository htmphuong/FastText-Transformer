# javalang
import functools
import operator

import javalang
from javalang.ast import Node

# anytree
from anytree import AnyNode

# file processing
import os
import json
import shutil

import numpy as np

# deep learning
import keras

from keras.utils import np_utils, pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
import keras.backend as K

from gensim.models import FastText
from gensim.models import KeyedVectors


def get_token(node):
    """

    Args:
        node: A node of AST tree

    Returns: Token name of that node

    """

    # Initialize the "token" variable with null string value
    token = 'String'

    if isinstance(node, str):
        token = node

    # If reviewing node is a set, return as 'Modifier'
    elif isinstance(node, set):
        token = 'Modifier'

    # If reviewing node is a Node, or we can say it is a class, return its class name
    elif isinstance(node, Node):
        token = node.__class__.__name__

    return token


def get_child(root):
    """

    Args:
        root: root node that we start to expand tree from

    Returns: list of child node of that root

    """

    # If root is a Node, initialize the "children" variable as children attribute of the root
    if isinstance(root, Node):
        children = root.children

    # If root is a set, convert root into list and initialize the "children" variable as that value
    elif isinstance(root, set):
        children = list(root)


    # Else, initialize children as an empty list
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))


def get_sequence(node, sequence):
    # this function is in order to get all tokens of the tree with root is "node"

    token, children = get_token(node), get_child(node)
    sequence.append(token)

    for child in children:
        get_sequence(child, sequence)


def get_ast(file):
    # Get the AST from file
    programFile = open(file, encoding="utf-8")

    # Extract text from opened file
    programText = programFile.read()

    try:
        # Tokenize extracted code using javalang library
        programTokens = javalang.tokenizer.tokenize(programText)

        # Create AST Parser and then parse tokenized code
        parser = javalang.parse.Parser(programTokens)
        programAst = parser.parse_member_declaration()

    except:
        programFile.close()

    # Close file
    programFile.close()

    return programAst


def get_all_ast(folder, dest):
    corpus = []
    print(folder)
    for file in os.listdir(folder):
        try:
            corpus.append(get_ast(os.path.join(folder, file)))
        except Exception as e:
            print(file, e)
            shutil.move(os.path.join(folder, file), os.path.join(dest, file))
    return corpus


def get_tokens_corpus(corpus):
    allTokens = []
    for ast in corpus:
        get_sequence(ast, allTokens)

    return allTokens


def create_tree(root, node, nodelist, parent=None):
    # Check number of node in node list
    id = len(nodelist)

    # Get token name and child nodes of "node"
    token, children = get_token(node), get_child(node)

    # If node list is empty
    if id == 0:
        # Define token name of root node
        root.token = token

        # Define data of root node
        root.data = node

        # Append root node to node list
        nodelist.append(root)

    # If the tree has already had root
    else:
        # Create new node with id, token name, data, parent node
        newNode = AnyNode(id=id, token=token, data=node, parent=parent)

        # Append new node to node list
        nodelist.append(newNode)

    # Loop through all child nodes of the root node
    for child in children:

        # If nodelist is empty, create new tree with parent node is root node
        if id == 0:
            create_tree(root, child, nodelist, parent=root)

        # Else, continue to expand tree with parent node is latest created node
        else:
            create_tree(root, child, nodelist, parent=newNode)


def get_node_and_edge(node, nodeIndexList, vocabDict, src, target):
    """
    Print out all the edges, represented by the node id. The two lists correspond one by one.
    The ID corresponding to the current node in vocabDict.
    """
    token = node.token
    nodeIndexList.append([vocabDict[token]])
    for child in node.children:
        src.append(node.id)
        target.append(child.id)
        get_node_and_edge(child, nodeIndexList, vocabDict, src, target)


def get_edge_flow(node, vocabDict, src, target):
    token = node.token
    if token == 'WhileStatement' or token == 'ForStatement':
        src.append(node.children[0].id)
        target.append(node.children[1].id)
    if token == 'IfStatement':
        src.append(node.children[0].id)
        target.append(node.children[1].id)
        if len(node.children) == 3:
            src.append(node.children[0].id)
            target.append(node.children[2].id)

    for child in node.children:
        get_edge_flow(child, vocabDict, src, target)


def get_edge_next_stmt(node, vocabDict, src, target):
    token = node.token
    if token == 'BlockStatement':
        for i in range(len(node.children) - 1):
            src.append(node.children[i].id)
            target.append(node.children[i + 1].id)
    for child in node.children:
        get_edge_next_stmt(child, vocabDict, src, target)


def cbow(idToVocab, total_vocab, edgeIndexesDict):
    for i in range(len(idToVocab)):
        context_node = []
        target = []

        if i in edgeIndexesDict:
            for v in edgeIndexesDict[i]:
                context_node.append(idToVocab[v])

        for src in edgeIndexesDict:
            if i in edgeIndexesDict[src]:
                context_node.append(idToVocab[src])

        target.append(idToVocab[i])

        context_node = [context_node]
        contextual = pad_sequences(context_node, 100)

        final_target = np_utils.to_categorical(target, total_vocab + 1)
        yield (contextual, final_target)


def Model(total_vocab, corpus, idToVocabCorpus):
    model = Sequential()
    model.add(Embedding(input_dim=total_vocab + 1, output_dim=100, input_length=100))
    model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(100,)))
    model.add(Dense(total_vocab + 1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    for i in range(1):
        cost = 0
        for j in range(len(corpus)):
            for contextual, final_target in cbow(idToVocabCorpus[j], total_vocab, corpus[j]):
                cost += model.train_on_batch(contextual, final_target)
            print(f"Doc {j + 1}: {cost}")
        print(f"Loop {i}: {cost}")

    return model


def get_edges_dict(file):
    global vocabDict
    programAst = get_ast(file)

    # Create new root
    nodeList = []
    newTree = AnyNode(id=0, token=None, data=None)
    create_tree(newTree, programAst, nodeList)

    nodeIndexList = []
    nodeSrc = []
    nodeTarg = []

    get_node_and_edge(newTree, nodeIndexList, vocabDict, nodeSrc, nodeTarg)
    get_edge_flow(newTree, vocabDict, nodeSrc, nodeTarg)
    get_edge_next_stmt(newTree, vocabDict, nodeSrc, nodeTarg)

    edgeIndexes = {}
    for src, tar in zip(nodeSrc, nodeTarg):
        if src not in edgeIndexes:
            edgeIndexes[src] = list()
        edgeIndexes[src].append(tar)

    idToVocab = {}
    for node in nodeList:
        if node.id not in idToVocab:
            idToVocab[node.id] = vocabDict[node.token]

    return edgeIndexes, idToVocab


def get_edges_dicts_corpus(folder):
    corpus = []
    idToVocabCorpus = []

    max_len = 0
    for file in os.listdir(folder):
        c, i = get_edges_dict(os.path.join(folder, file))
        if len(i) > max_len:
            max_len = len(i)
        corpus.append(c)
        idToVocabCorpus.append(i)
    return corpus, idToVocabCorpus, max_len


def get_embedding_vectors(modelPath, vocabDict, dest, dimension):
    model = keras.models.load_model(modelPath)
    weights = model.get_weights()[0]

    vect_file = open(dest, 'w+', encoding="utf-8")
    vect_file.write('{} {}\n'.format(len(vocabDict), dimension))

    for k, v in vocabDict.items():
        final_vec = ' '.join(map(str, list(weights[v, :])))
        vect_file.write('{} {} {}\n'.format(v, k, final_vec))

    vect_file.close()

    return weights


def get_embedding_vectors_gensim(modelPath, vocabDict, dest, dimension):
    model = FastText.load(modelPath)
    weights = model.wv

    vect_file = open(dest, 'w+', encoding="utf-8")
    vect_file.write('{} {}\n'.format(len(vocabDict), dimension))

    for k, v in vocabDict.items():
        final_vec = ' '.join(map(str, list(weights[k])))
        vect_file.write('{} {} {}\n'.format(v, k, final_vec))

    vect_file.close()

    return weights


# def embedding_ast(file, embeddingVects):
#     edIdx, idToVocab = get_edges_dict(file)
#
#     new_vec = []
#     for i in range(len(idToVocab)):
#         new_vec.append(embeddingVects[idToVocab[i], :].tolist())
#
#     new_vec = np.array(new_vec)
#     new_vec = np.pad(new_vec, ((0, 3000 - new_vec.shape[0]), (0, 0)), 'constant')
#     new_vec = new_vec.tolist()
#     return new_vec

def embedding_ast(file, embeddingVects, vocabDict, max_len):
    edIdx, idToVocab = get_edges_dict(file)

    new_vec = []
    for i in range(len(idToVocab)):
        new_vec.append(embeddingVects[list(vocabDict.keys())[list(vocabDict.values()).index(idToVocab[i])]].tolist())

    new_vec = np.array(new_vec)
    new_vec = np.pad(new_vec, ((0, max_len - new_vec.shape[0]), (0, 0)), 'constant')
    new_vec = new_vec.tolist()
    return new_vec


if __name__ == "__main__":
    dataset = "lucene"

    root_folder = "./data/Clean Data/" + dataset

    finalAllTokens = []

    for fol in os.listdir(root_folder):
        source = root_folder + "/" + fol
        dest = "./data/Have bug/" + dataset + "/" + fol

        if not os.path.exists(dest):
            os.makedirs(dest, mode=0o777)
            print("Create", dest)

        allTokens = get_tokens_corpus(get_all_ast(source, dest))

        finalAllTokens.append(allTokens[:])

    finalAllTokens = functools.reduce(operator.iconcat, finalAllTokens, [])

    allTokens = list(set(finalAllTokens))
    vocabSize = len(allTokens)
    tokenIds = range(1, vocabSize + 1)
    vocabDict = dict(zip(allTokens, tokenIds))

    print(vocabDict)

    with open("./data/vocab_" + dataset + ".txt", "w+", encoding="utf8") as f:
        json.dump(vocabDict, f)

    # Load vocab
    # with open("./data/vocab_" + dataset + ".txt", encoding="utf8") as f:
    #     vocabDict = json.load(f)

    # finalCorpus = []
    # finalIdtoVocabCorpus = []
    #
    # for fol in os.listdir(root_folder):
    #     source = root_folder + "/" + fol
    #     corpus, idToVocabCorpus = get_edges_dicts_corpus(source)
    #     finalCorpus.append(corpus[:])
    #     finalIdtoVocabCorpus.append(idToVocabCorpus[:])
    #
    # finalCorpus = functools.reduce(operator.iconcat, finalCorpus, [])
    # finalIdtoVocabCorpus = functools.reduce(operator.iconcat, finalIdtoVocabCorpus, [])
    #
    # model = Model(len(vocabDict), finalCorpus, finalIdtoVocabCorpus)
    # modelPath = './model/saved_cbow_model_' + dataset
    # model.save(modelPath)

    max_len = 0
    for fol in os.listdir(root_folder):
        source = root_folder + "/" + fol
        corpus, idToVocabCorpus, max_len_ = get_edges_dicts_corpus(source)
        if max_len_ > max_len:
            max_len = max_len_

    print(max_len)

    model = FastText(vector_size=100, window=3, min_count=1)  # instantiate
    model.build_vocab(corpus_iterable=allTokens)
    model.train(corpus_iterable=allTokens, total_examples=len(allTokens), epochs=10)

    word_vectors = model.wv
    word_vectors.save("./model/gensim_fasttext_"+dataset+".wordvectors")

    # Load back with memory-mapping = read-only, shared across processes.
    embeddingVects = KeyedVectors.load("./model/gensim_fasttext_"+dataset+".wordvectors", mmap='r')
    # embeddingVects = get_embedding_vectors(modelPath, vocabDict, dest, dimension=100)

    for dir in os.listdir(root_folder):
        X = []
        for f in os.listdir(os.path.join(root_folder, dir)):
            X.append(embedding_ast(os.path.join(root_folder, dir + "/" + f), embeddingVects, vocabDict, max_len))

        X = np.array(X)

        arr_reshaped = X.reshape(X.shape[0], -1)

        # saving reshaped array to file.
        np.savetxt("./data/X_" + dataset + "_" + dir + ".txt", arr_reshaped)

        print(X.shape)
