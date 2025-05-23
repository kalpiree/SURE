# import sys
# import copy
# import random
# import numpy as np
# from tqdm import tqdm
# from collections import defaultdict
# from multiprocessing import Process, Queue

# def random_neq(l, r, s):
#     t = np.random.randint(l, r)
#     while t in s:
#         t = np.random.randint(l, r)
#     return t

# def computeRePos(time_seq, time_span):
    
#     size = time_seq.shape[0]
#     time_matrix = np.zeros([size, size], dtype=np.int32)
#     for i in range(size):
#         for j in range(size):
#             span = abs(time_seq[i]-time_seq[j])
#             if span > time_span:
#                 time_matrix[i][j] = time_span
#             else:
#                 time_matrix[i][j] = span
#     return time_matrix

# def Relation(user_train, usernum, maxlen, time_span):
#     data_train = dict()
#     for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
#         time_seq = np.zeros([maxlen], dtype=np.int32)
#         idx = maxlen - 1
#         for i in reversed(user_train[user][:-1]):
#             time_seq[idx] = i[1]
#             idx -= 1
#             if idx == -1: break
#         data_train[user] = computeRePos(time_seq, time_span)
#     return data_train

# def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, result_queue, SEED):
#     def sample(user):

#         seq = np.zeros([maxlen], dtype=np.int32)
#         time_seq = np.zeros([maxlen], dtype=np.int32)
#         pos = np.zeros([maxlen], dtype=np.int32)
#         neg = np.zeros([maxlen], dtype=np.int32)
#         nxt = user_train[user][-1][0]
    
#         idx = maxlen - 1
#         ts = set(map(lambda x: x[0],user_train[user]))
#         for i in reversed(user_train[user][:-1]):
#             seq[idx] = i[0]
#             time_seq[idx] = i[1]
#             pos[idx] = nxt
#             if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
#             nxt = i[0]
#             idx -= 1
#             if idx == -1: break
#         time_matrix = relation_matrix[user]
#         return (user, seq, time_seq, time_matrix, pos, neg)

#     np.random.seed(SEED)
#     while True:
#         one_batch = []
#         for i in range(batch_size):
#             user = np.random.randint(1, usernum + 1)
#             while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
#             one_batch.append(sample(user))

#         result_queue.put(zip(*one_batch))

# class WarpSampler(object):
#     def __init__(self, User, usernum, itemnum, relation_matrix, batch_size=64, maxlen=10,n_workers=1):
#         self.result_queue = Queue(maxsize=n_workers * 10)
#         self.processors = []
#         for i in range(n_workers):
#             self.processors.append(
#                 Process(target=sample_function, args=(User,
#                                                       usernum,
#                                                       itemnum,
#                                                       batch_size,
#                                                       maxlen,
#                                                       relation_matrix,
#                                                       self.result_queue,
#                                                       np.random.randint(2e9)
#                                                       )))
#             self.processors[-1].daemon = True
#             self.processors[-1].start()

#     def next_batch(self):
#         return self.result_queue.get()

#     def close(self):
#         for p in self.processors:
#             p.terminate()
#             p.join()

# def timeSlice(time_set):
#     time_min = min(time_set)
#     time_map = dict()
#     for time in time_set: # float as map key?
#         time_map[time] = int(round(float(time-time_min)))
#     return time_map

# def cleanAndsort(User, time_map):
#     User_filted = dict()
#     user_set = set()
#     item_set = set()
#     for user, items in User.items():
#         user_set.add(user)
#         User_filted[user] = items
#         for item in items:
#             item_set.add(item[0])
#     user_map = dict()
#     item_map = dict()
#     for u, user in enumerate(user_set):
#         user_map[user] = u+1
#     for i, item in enumerate(item_set):
#         item_map[item] = i+1
    
#     for user, items in User_filted.items():
#         User_filted[user] = sorted(items, key=lambda x: x[1])

#     User_res = dict()
#     for user, items in User_filted.items():
#         User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]]], items))

#     time_max = set()
#     for user, items in User_res.items():
#         time_list = list(map(lambda x: x[1], items))
#         time_diff = set()
#         for i in range(len(time_list)-1):
#             if time_list[i+1]-time_list[i] != 0:
#                 time_diff.add(time_list[i+1]-time_list[i])
#         if len(time_diff)==0:
#             time_scale = 1
#         else:
#             time_scale = min(time_diff)
#         time_min = min(time_list)
#         User_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1)], items))
#         time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

#     return User_res, len(user_set), len(item_set), max(time_max)

# def data_partition(fname):
#     usernum = 0
#     itemnum = 0
#     User = defaultdict(list)
#     user_train = {}
#     user_valid = {}
#     user_test = {}
    
#     print('Preparing data...')
#     f = open('data/%s.txt' % fname, 'r')
#     time_set = set()

#     user_count = defaultdict(int)
#     item_count = defaultdict(int)
#     for line in f:
#         try:
#             u, i, rating, timestamp = line.rstrip().split('\t')
#         except:
#             u, i, timestamp = line.rstrip().split('\t')
#         u = int(u)
#         i = int(i)
#         user_count[u]+=1
#         item_count[i]+=1
#     f.close()
#     f = open('data/%s.txt' % fname, 'r') # try?...ugly data pre-processing code
#     for line in f:
#         try:
#             u, i, rating, timestamp = line.rstrip().split('\t')
#         except:
#             u, i, timestamp = line.rstrip().split('\t')
#         u = int(u)
#         i = int(i)
#         timestamp = float(timestamp)
#         if user_count[u]<5 or item_count[i]<5: # hard-coded
#             continue
#         time_set.add(timestamp)
#         User[u].append([i, timestamp])
#     f.close()
#     time_map = timeSlice(time_set)
#     User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

#     for user in User:
#         nfeedback = len(User[user])
#         if nfeedback < 3:
#             user_train[user] = User[user]
#             user_valid[user] = []
#             user_test[user] = []
#         else:
#             user_train[user] = User[user][:-2]
#             user_valid[user] = []
#             user_valid[user].append(User[user][-2])
#             user_test[user] = []
#             user_test[user].append(User[user][-1])
#     print('Preparing done...')
#     return [user_train, user_valid, user_test, usernum, itemnum, timenum]


# def evaluate(model, dataset, args):
#     [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

#     NDCG = 0.0
#     HT = 0.0
#     valid_user = 0.0

#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:

#         if len(train[u]) < 1 or len(test[u]) < 1: continue

#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         time_seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
        
#         seq[idx] = valid[u][0][0]
#         time_seq[idx] = valid[u][0][1]
#         idx -= 1
#         for i in reversed(train[u]):
#             seq[idx] = i[0]
#             time_seq[idx] = i[1]
#             idx -= 1
#             if idx == -1: break
#         rated = set(map(lambda x: x[0],train[u]))
#         rated.add(valid[u][0][0])
#         rated.add(test[u][0][0])
#         rated.add(0)
#         item_idx = [test[u][0][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)

#         time_matrix = computeRePos(time_seq, args.time_span)

#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix],item_idx]])
#         predictions = predictions[0]

#         rank = predictions.argsort().argsort()[0].item()

#         valid_user += 1

#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.',end='')
#             sys.stdout.flush()

#     return NDCG / valid_user, HT / valid_user


# def evaluate_valid(model, dataset, args):
#     [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < 1: continue

#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         time_seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i[0]
#             time_seq[idx] = i[1]
#             idx -= 1
#             if idx == -1: break

#         rated = set(map(lambda x: x[0], train[u]))
#         rated.add(valid[u][0][0])
#         rated.add(0)
#         item_idx = [valid[u][0][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)

#         time_matrix = computeRePos(time_seq, args.time_span)
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix],item_idx]])
#         predictions = predictions[0]

#         rank = predictions.argsort().argsort()[0].item()

#         valid_user += 1

#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.',end='')
#             sys.stdout.flush()

#     return NDCG / valid_user, HT / valid_user



import sys
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue

# ======== Utility Functions ========

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def computeRePos(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            time_matrix[i][j] = min(span, time_span)
    return time_matrix

# ======== Relation Matrix ========

def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(user_train.keys(), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for item in reversed(user_train[user][:-1]):
            time_seq[idx] = 0  # no real timestamp, dummy 0
            idx -= 1
            if idx == -1:
                break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train

# ======== Sampler ========

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, result_queue, SEED):
    user_list = list(user_train.keys())

    def sample(user):
        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train[user][-1]

        idx = maxlen - 1
        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            time_seq[idx] = 0  # dummy
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        time_matrix = relation_matrix[user]
        return (user, seq, time_seq, time_matrix, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        while len(one_batch) < batch_size:
            user = random.choice(user_list)
            if len(user_train[user]) <= 1:
                continue
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User, usernum, itemnum, batch_size, maxlen, relation_matrix, self.result_queue, np.random.randint(2e9)))
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# ======== Phase-wise Data Loading ========

def data_partition_phase(train_file, eval_file):
    user_train = dict()
    user_test = dict()
    user_set = set()
    item_set = set()

    print('Loading training data from:', train_file)
    with open(train_file, 'r') as f:
        for line in f:
            user_id, item_id = map(int, line.strip().split())
            if user_id not in user_train:
                user_train[user_id] = []
            user_train[user_id].append(item_id)
            user_set.add(user_id)
            item_set.add(item_id)

    print('Loading evaluation data from:', eval_file)
    eval_df = pd.read_csv(eval_file)

    for idx, row in eval_df.iterrows():
        user_id = int(row['user_idx'])
        true_item = int(row['true_item'])
        candidate_items = eval(row['candidate_items'])  # careful: parse as list
        user_test[user_id] = (true_item, candidate_items)
        item_set.update(candidate_items)

    usernum = max(user_set) + 1
    itemnum = max(item_set) + 1

    print('Loaded:', len(user_train), 'users for training,', len(user_test), 'users for testing')
    return user_train, user_test, usernum, itemnum

# ======== Evaluation ========

# def evaluate_phase(model, user_train, user_test, args, top_k=10):
#     NDCG = 0.0
#     HT = 0.0
#     valid_user = 0.0

#     users = list(user_test.keys())
#     for u in users:
#         if u not in user_train or len(user_train[u]) == 0:
#             continue

#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1

#         for i in reversed(user_train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1:
#                 break

#         true_item, candidate_items = user_test[u]

#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [candidate_items]]])
#         predictions = predictions[0]

#         rank = predictions.argsort().argsort()[0].item()

#         valid_user += 1
#         if rank < top_k:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1

#     return NDCG / valid_user, HT / valid_user

# def evaluate_phase(model, user_train, user_test, args, top_k=10):
#     NDCG = 0.0
#     HT = 0.0
#     valid_user = 0.0

#     users = list(user_test.keys())
#     for u in users:
#         if u not in user_train or len(user_train[u]) == 0:
#             continue

#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1

#         for i in reversed(user_train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1:
#                 break

#         true_item, candidate_items = user_test[u]

#         dummy_time_matrix = np.zeros((1, args.maxlen, args.maxlen), dtype=np.int32)  # <-- corrected here!

#         predictions = -model.predict(
#             np.array([u]),
#             np.array([seq]),
#             dummy_time_matrix,
#             np.array(candidate_items)
#         )
#         predictions = predictions[0]

#         rank = predictions.argsort().argsort()[0].item()

#         valid_user += 1
#         if rank < top_k:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1

#     return NDCG / valid_user, HT / valid_user

def evaluate_phase(model, user_train, user_test, args, top_k=10):
    NDCG = 0.0
    HT = 0.0
    MRR = 0.0
    valid_user = 0.0

    users = list(user_test.keys())
    for u in users:
        if u not in user_train or len(user_train[u]) == 0:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        for i in reversed(user_train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        true_item, candidate_items = user_test[u]

        # Dummy time matrix because we have no real timestamps
        time_matrix_dummy = np.zeros((1, args.maxlen, args.maxlen), dtype=np.int32)

        predictions = -model.predict(np.array([u]), np.array([seq]), time_matrix_dummy, np.array(candidate_items))
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < top_k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        MRR += 1.0 / (rank + 1)

    return NDCG / valid_user, HT / valid_user, MRR / valid_user




