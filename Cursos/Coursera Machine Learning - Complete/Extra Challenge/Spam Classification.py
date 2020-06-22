
import os
import numpy as np
from sklearn import svm, preprocessing
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.corpus import brown
from nltk import FreqDist
import re
from string import punctuation



# cleans up the email
def processEmail(email, path):

    # open file location
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emfile = os.path.join(dir_path, path, email)

    # verifies for non utf8 characters.
    weird_enc = 0
    try:
        emfile = open(emfile, 'r', encoding = 'utf8')
        emStr = ''.join(emfile.readlines())
        emfile.close()
    except:
        emfile = os.path.join(dir_path, path, email)
        emfile = open(emfile, 'r', encoding = 'latin1')
        emStr = ''.join(emfile.readlines())
        emfile.close()
        weird_enc = 1
        
    # lower case email
    emStr = emStr.lower()

    # handle HTML
    emStr = re.sub(r'<[^<>]+>', ' Html ', emStr)

    # handle numbers = 'number'
    emStr = re.sub(r'[0-9]+', ' Number ', emStr)

    # handle links = 'httpaddr'
    emStr = re.sub(r'(http|https)://[^\s]*', ' Httpaddr ', emStr)

    # handle emails = 'emailaddr'
    emStr = re.sub(r'[^\s]+@[^\s]+', ' Emailaddr ', emStr)

    # handle dollar sign = 'dollar'
    emStr = re.sub(r'[$]+', ' Dollar ', emStr)

    # handle punctuation, symbols and \n
    emStr = re.sub(r'[.,:;\]\[\\\(\)\-\+\{\}\|\_/%@"\'*>\=?!]', '', emStr)
    emStr = re.sub(r'[^\x20-\x7E]', '', emStr)
    emStr = re.sub(r'[\n]', '', emStr)
    
    # filter useless spaces and short words
    emLis = emStr.split(' ')
    emLis = list(filter(lambda a: a != '', emLis))
    emLis = list(filter(lambda a: len(a) > 2, emLis))

    # handle weird chars:
    if weird_enc == 1:
        emLis.append("Enc")

    # returns list of a cleaned up version of the email
    return emLis

# creates the features vector for one example.
# the features are the amount of times the words in vocabLis are found in the email +
# any html, number, email address, http or dollar sign ($).
def makeFeatures(emLis, vocabLis):   
    featLis = [0 for i in range(len(vocabLis))]
    for i in range(6):
        featLis.append(0)

    for word in emLis:
        if word == 'Html':
            featLis[-1] = 1
        if word == 'Number':
            featLis[-2] = 1
        if word == 'Httpaddr':
            featLis[-3] = 1
        if word == 'Emailaddr':
            featLis[-4] = 1
        if word == 'Dollar':
            featLis[-5] = 1
        if word == 'Enc':
            featLis[-6] = 1

        if word in vocabLis:
            idx = vocabLis.index(word)
            featLis[idx] = 1

    return featLis

# Shouffles two lists of the same length, maintaining the pairing
def doubleShuffle(A, B):
    n = len(A)
    m = len(B)

    idx = np.asarray([i for i in range(n)])
    np.random.shuffle(idx)

    for i in range(n):
        A[i], A[idx[i]] = A[idx[i]], A[i]
        B[i], B[idx[i]] = B[idx[i]], B[i]  


def main():
    
    my_vocabLis = ['work', 'buy', 'sell', 'credit', 'card', 'number', 'pay', 'payment', 'meet', 'singles', 'babes', 'shopper',
                'shop', 'extra', 'earn', 'money', 'rich', 'income', 'making', 'make', 'earnings', 'potential',
                'cheap', 'check', 'cash', 'lowest', 'price', 'profits', 'profit', 'save', 'unsecured', 'debt', 'full',
                'refund', 'fund', 'stock', 'alert', 'limited', 'cost', 'costs', 'investment', 'initial', 'paid', 'get', 'freedom', 'password',
                'passwords', 'miracle', 'success', 'teen', 'wife', 'click', 'here', 'increase', 'sales', 'sale', 'member', 'notspam', 'junk', 'this',
                'isn`t', 'unsubscribe', 'spam', 'bald', 'baldness', 'viagra', 'sex', 'step', 'manage', 'team', 'build', 'car',
                'computer', 'try', 'tried', 'dude', 'man', 'pal', 'partner', 'night', 'day', 'date', 'walk', 'stay',
                'put', 'get', 'insecure', 'security', 'scam', 'not', 'exmh', 'meeting', 'message', 'mail', 'email', 'child',
                'friend', 'hello', 'hi', 'greetings', 'name', 'dont', 'isnt', 'don`t', 'made', 'maid', 'but', 'the', 'will',
                'market', 'marketing', 'stocking', 'group', 'meeting', 'boss', 'chairman', 'owner', 'enterprise', 'company',
                'just', 'sports', 'why', 'would', 'you', 'that', 'them', 'government', 'spy', 'crack', 'cracked', 'hacked',
                'account', 'book', 'address', 'advertisement', 'add', 'announcement', 'announce', 'announced', 'legal',
                'illegal', 'coin', 'bitcoin', 'bit', 'lotery', 'change', 'exchange', 'connect', 'crypt', 'crypto', 'encrypted',
                'currency', 'current', 'events', 'weather', 'nice', 'job', 'interview', 'task', 'leader', 'what', 'say',
                'ask', 'anything', 'someone', 'porn', 'tickets', 'movie', 'show', 'children', 'grandma', 'mom', 'dad', 'doctor',
                'hospital', 'bill', 'bils', 'surgery', 'salary', 'else', 'around', 'hack', 'witch', 'school', 'teach',
                'come', 'dump', 'file', 'files', 'donate', 'receive', 'take', 'future', 'past', 'transaction', 'finance', 'financial',
                'treasure', 'are', 'doin', 'doing', 'smh', 'lmfao', 'lmao', 'wtf', 'wth', 'ohh', 'yes', 'flow', 'workflow',
                'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'saturday', 'week', 'weekday', 'weekend', 'bank', 'shower', 'good',
                'aid', 'usd', 'china', 'russia', 'party', 'happy', 'hour', 'late', 'intel', 'intelligence', 'recruit', 'program', 'office',
                'safe', 'totally', 'accuracy', 'accurate', 'shit', 'join', 'venture', 'joint', 'human', 'source', 'trust',
                'trustworthy', 'worth', 'worthy', 'resource', 'protein', 'shake', 'health', 'insurance', 'save', 'lifetime', 'life',
                'lives', 'live', 'eye', 'for', 'forth', 'presence', 'recruiter', 'basketball', 'football', 'grain', 'lost', 'loss',
                'lose', 'loose', 'shoe', 'windows', 'mac', 'window', 'linux', 'another', 'king', 'queen', 'worthless', 'game', 'anime',
                'animation', 'film', 'pen', 'paper', 'pencil', 'eraser', 'word', 'power', 'point', 'excel', 'press', 'new', 'news', 'please',
                'dear', 'top', 'maintenance', 'away', 'give', 'gift', 'apply', 'today', 'tomorrow', 'mastercard', 'call', 'visa', 'express',
                'deliver', 'delivery', 'rescue', 'enhancement', 'male', 'female', 'cream', 'lotion', 'body', 'sexy', 'key', 'avoid',
                'paying', 'skin', 'healthy', 'beauty', 'placement', 'order', 'trick', 'perform', 'performance', 'demand', 'flop',
                'drop', 'lift', 'pharma', 'big', 'love', 'travel', 'hate', 'discover', 'journey', 'thanks', 'yours', 'truly', 'mister', 'miss',
                'model', 'models', 'secret', 'secrets', 'evolve', 'cry', 'deposit', 'fuck', 'shame', 'shameful', 'shameless', 'series',
                'off', 'output', 'send', 'against', 'again', 'restore', 'machine', 'learning', 'library', 'presentation',
                'iam', 'youre', 'your', 'ship', 'shipping', 'shipped', 'dig', 'dirt', 'own', 'home', 'homebased', 'sleep',
                'cent', 'cents', 'dollar', 'discount', 'promotion', 'want', 'fast', 'loan', 'loans', 'rate', 'mortage', 'rates',
                'quote', 'serious', 'statement', 'consolidate', 'low', 'lower', 'fire', 'fired', 'should', 'leave', 'left', 'right',
                'satisfaction', 'engine', 'search', 'research', 'form', 'follow', 'recipient', 'receipt', 'visit', 'website', 'believe',
                'weight', 'vicodin', 'xanax', 'billion', 'million', 'millionaire', 'hundred', 'percent', 'guaranteed', 'guarantee', 'thousands',
                'long', 'distance', 'short', 'small', 'gimmick', 'questions', 'question', 'prize', 'prizes', 'chances', 'trial', 'honor', 'winner',
                'fax', 'dvd', 'leads', 'risk', 'delete', 'instant', 'limited', 'last', 'iphone', 'android', 'refrigerator', 'deluxe',
                'platinum', 'rolex', 'luxury', 'ebook', 'apple', 'cellphone', 'urgent', 'congratulations', 'lesson', 'trick', 'one', 'only',
                'cryptocurrency']
    
    vocabLis = FreqDist([word.lower() for word in brown.words()]).most_common(1000)
    vocabLis[:] = [x[0] for x in vocabLis if x[0] not in [punc for punc in punctuation]][:500] + my_vocabLis
    
    X = []
    y = []

    print("-- Preprocessing text files --\n")

    # Creates the feature matrix and labels each example, acording to the folder they were found in (Spam or NoSpam).
    for folder in os.listdir(os.getcwd()):
        if folder == "NoSpam":
            for file in os.listdir(os.path.join(os.getcwd(), 'NoSpam')):
                processed = processEmail(file, 'NoSpam')
                Xn = makeFeatures(processed, vocabLis)
                X.append(Xn)
                y.append(0)
        if folder == "Spam":
            for file in os.listdir(os.path.join(os.getcwd(), 'Spam')):
                processed = processEmail(file, 'Spam')
                Xn = makeFeatures(processed, vocabLis)
                X.append(Xn)
                y.append(1)

    # randomizes X and y
    doubleShuffle(X, y)

    cut = int(len(X)*0.2)

    # creates train set and test set
    train_X = X[cut:]
    train_y = y[cut:]

    test_X = X[:cut]
    test_y = y[:cut]

    print("-- Training SVM --\n")

    # Trains an SVM with Sklearn to decide if an email is or isnt spam.
    clf = svm.SVC()
    clf.fit(train_X, train_y)

    preds = clf.predict(test_X)    

    print(classification_report(test_y, preds, target_names = ['Not spam', 'Spam']))
    print('Accuracy: ', accuracy_score(test_y, preds), '\n')


    # Performs PCA for data visualization
    pca = PCA(n_components = 2)
    pca.fit(X)

    X = pca.transform(X)

    plt.scatter(X[:, 0], X[:, 1], c = y)
    plt.show()

main()







