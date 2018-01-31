# coding=utf-8
import multiprocessing
import tensorflow as tf
from math import sqrt
import numpy as np

cores = multiprocessing.cpu_count()

Embedding_size = 1000  # latent factor size
User_number = 1000  # user number of data set
POI_number = 1000  # POI number of data set
Negative_sample = 10  # not too big
POIs = set(range(POI_number))
workdir = 'data/'  # your data directory
NEG_SAMPLE_FILE = workdir + 'neg_sample_train.txt'

learning_rate_value = 0.1  # Initial learning rate
learning_rate = tf.Variable(float(learning_rate_value),
                            trainable=False,
                            dtype=tf.float32)


class Discriminator():
    def __init__(self,
                 poiNumber,
                 userNumber,
                 embedding_size,
                 lamda,
                 param=None,
                 initdelta=1,
                 learning_rate=0.1):

        self.poiNumber = poiNumber
        self.userNumber = userNumber
        self.embedding_size = embedding_size
        self.lamda = lamda
        self.embedding_parameter = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.discriminator_parameters = []

        with tf.variable_scope('discriminator'):
            if self.embedding_parameter is None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNumber, self.embedding_size],
                                      minval=-self.initdelta,
                                      maxval=self.initdelta,
                                      dtype=tf.float32))
                self.poi_embeddings = tf.Variable(
                    tf.random_uniform([self.poiNumber, self.embedding_size],
                                      minval=-self.initdelta,
                                      maxval=self.initdelta,
                                      dtype=tf.float32))
                self.bias = tf.Variable(tf.zeros([self.poiNumber]))
                # self.user_embeddings_2 = tf.Variable(
                #     tf.random_uniform([self.userNum, self.emb_dim],
                #                       minval=-self.initdelta,
                #                       maxval=self.initdelta,
                #                       dtype=tf.float32))
            else:
                self.user_embeddings = tf.Variable(self.embedding_parameter[0])
                self.poi_embeddings = tf.Variable(self.embedding_parameter[1])
                self.bias = tf.Variable(self.embedding_parameter[2])

        self.discriminator_parameters = [self.user_embeddings, self.poi_embeddings, self.bias]

        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.pos = tf.placeholder(tf.int32)
        self.neg = tf.placeholder(tf.int32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.pos_embedding = tf.nn.embedding_lookup(self.poi_embeddings, self.pos)
        self.pos_bias = tf.gather(self.bias, self.pos)
        self.neg_embedding = tf.nn.embedding_lookup(self.poi_embeddings, self.neg)
        self.neg_bias = tf.gather(self.bias, self.neg)

        # ***************************************************************************
        # self.pre_logits_pos = tf.sigmoid(
        #    tf.reduce_sum(tf.multiply(self.u_embedding, self.pos_embedding), 1) + self.pos_bias)

        # self.pre_logits_neg = tf.sigmoid(
        #    tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_embedding), 1) + self.neg_bias)


        # self.pre_loss = -(tf.reduce_mean(tf.log(self.pre_logits_pos)) + tf.reduce_mean(tf.log(1 - self.pre_logits_neg))
        #                  + self.lamda * (
        #                      tf.nn.l2_loss(self.u_embedding) +
        #                     tf.nn.l2_loss(self.pos_embedding) +
        #                    tf.nn.l2_loss(self.pos_bias) +
        #                   tf.nn.l2_loss(self.neg_embedding) +
        #                  tf.nn.l2_loss(self.neg_bias)
        #             ))
        # ***************************************************************************

        self.logits = tf.sigmoid(
            tf.reduce_sum(tf.multiply(self.u_embedding, self.pos_embedding - self.neg_embedding),
                          1) + self.pos_bias - self.neg_bias)
        self.loss = -tf.reduce_mean(tf.log(self.logits)) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) +
            tf.nn.l2_loss(self.pos_embedding) +
            tf.nn.l2_loss(self.pos_bias) +
            tf.nn.l2_loss(self.neg_embedding) +
            tf.nn.l2_loss(self.neg_bias)
        )

        D_train_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.D_train_step = D_train_optimizer.minimize(self.loss,
                                                       var_list=self.discriminator_parameters)

        self.all_poi_score = tf.matmul(self.u_embedding,
                                       self.poi_embeddings,
                                       transpose_a=False,
                                       transpose_b=True) + self.bias

        self.logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.poi_embeddings), 1) + self.bias

        # for negative sample
        self.neg_sample_score = tf.reduce_sum(tf.multiply(self.u_embedding, self.poi_embeddings), 1) + self.bias

        # self.all_poi_score = tf.matmul(self.u_embedding, self.poi_embeddings, transpose_a=False,
        #                                transpose_b=True) + self.bias


user_pos_train = {}
with open(workdir + 'train.txt')as fin:  # train set
    for line in fin:  # user_pos_train[uer id] = [lid1, lid2, lid3, ...]
        line = line.split()
        uid = int(line[0])
        lid = int(line[1])
        r = float(line[2])
        if r > 0:
            if uid in user_pos_train:
                user_pos_train[uid].append(lid)
            else:
                user_pos_train[uid] = [lid]

user_pos_test = {}
with open(workdir + 'test.txt')as fin:  # test set
    for line in fin:  # user_pos_test[uer id] = [lid1, lid2, lid3, ...]
        line = line.split()
        uid = int(line[0])
        lid = int(line[1])
        r = float(line[2])
        if r > 0:
            if uid in user_pos_test:
                user_pos_test[uid].append(lid)
            else:
                user_pos_test[uid] = [lid]

all_users = list(user_pos_train.keys())
all_users.sort()


def get_poi_by_poi_Matrix():
    p_p_M = np.full((POI_number, POI_number), 0, dtype=np.float32)
    p_p_dict = dict()
    print 'p_p_M construct start'
    poi_file = open('./data_Foursquare/Foursquare_poi_coos.txt', 'r')
    file_list = poi_file.readlines()
    for line in file_list:
        line_split = line.split()
        poi, x, y = line_split[0], line_split[1], line_split[2]
        poi, x, y = int(poi), float(x), float(y)
        p_p_dict[poi] = (x, y)  # dict[int] = (float, float)
    for i in range(POI_number):
        i_x, i_y = p_p_dict[i]
        for j in range(i, POI_number):
            j_x, j_y = p_p_dict[j]
            distance = sqrt((i_x - j_x) ** 2 + (i_y - j_y) ** 2)
            if 0 < distance <= 0.1:
                p_p_M[i, j] = 1 / (0.5 + distance)
                p_p_M[j, i] = 1 / (0.5 + distance)
    poi_file.close()
    print 'p_p_M construct done'
    return p_p_M


def get_user_poi_and_friens_poi(user):
    # poi_loc = dict()
    # poi_file = open('./data_Gowalla/Gowalla_poi_coos.txt', 'r')
    # file_list = poi_file.readlines()
    # for line in file_list:
    #     line_split = line.split()
    #     poi, x, y = line_split[0], line_split[1], line_split[2]
    #     poi, x, y = int(poi), float(x), float(y)
    #     poi_loc[poi] = (x, y)  # dict[int] = (float, float)
    # poi_file.close()
    friends = []
    with open('./data_Gowalla/Gowalla_social_relations.txt', 'r') as file:
        for line in file:
            line_split = line.split()
            if int(line_split[0]) == user or int(line_split[1]) == user:
                friends.append(int(line_split[0]))
                friends.append(int(line_split[1]))
        friends = list(set(friends))
        friends.remove(user)
    poi_list = []
    friends_poi_list = []
    with open('./data_Gowalla/Gowalla_train.txt', 'r') as file:
        for line in file:
            line_split = line.split()
            if int(line_split[0]) == user:
                poi_list.append(int(line_split[1]))
            elif int(line_split[0]) in friends:
                friends_poi_list.append(int(line_split[1]))
    return friends_poi_list, poi_list
# def get_user_poi_and_friens_poi(user):
#     poi_list = user_pos_train[user]
    # friends_list = user_friends[user]
    # friends_poi_list = []
    # for friend in friends_list:
    #     friends_poi_list.extend(user_pos_train[friend])
    # return friends_poi_list, poi_list
    # return poi_list


#                     0.2    0.1
def get_reward(user, alpha, beta, p_p_M):  # alpha&beta control the weight of rewards
    # geographical    与该user去过的点的地理位置近
    # social   该user的朋友也去过
    # the_user_pos_poi = user_pos_train[user]
    reward = [0.7] * POI_number
    geo_reward = [0] * POI_number
    friends_poi_list, poi_list = get_user_poi_and_friens_poi(user)
    poi_list = get_user_poi_and_friens_poi(user)
    for each_poi in poi_list:
        row_result = p_p_M[each_poi]
        geo_reward += row_result
    for ii in range(POI_number):
        if ii in friends_poi_list:
            reward[ii] += (1 * beta)
        if geo_reward[ii] > 10:
            geo_reward[ii] = 10

    reward += (geo_reward * alpha)
    # reward += (((geo_reward * alpha) - 0.2) * 0.1 + 0.2)

    return reward


def generate_dns(sess, model, filename, p_p_M):
    data = []
    for user in user_pos_train:
        reward = get_reward(user, 0.2, 0.1, p_p_M)
        pos = user_pos_train[user]
        dns_rating = sess.run(model.dns_rating, {model.u: user})
        dns_rating = np.multiply(np.array(dns_rating), np.array(reward))
        neg = []
        candidates = list(POIs - set(pos))

        # flag = True
        # lens = 0
        # while flag:
        #     if
        #     if lens == len(pos):
        #         flag = False

        for _ in range(len(pos)):
            choice = np.random.choice(candidates, Negative_sample)
            choice_score = dns_rating[choice]
            neg.append(choice[np.argmax(choice_score)])

        for i in range(len(pos)):
            data.append(str(user) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


# metrics
# def dcg_at_k(r, k):
#     r = np.asfarray(r)[:k]
#     return np.sum(r / np.log2(np.arange(2, r.size + 2)))
# def pre():
#     return
#
# def recall():
#     return

def ndcg_at_k(r, k):
    idcg = 1.0
    dcg = float(r[0])
    for i, p in enumerate(r[1:k]):
        if p == 1:
            dcg += 1.0 / np.log(i + 2)
        idcg += 1.0 / np.log(i + 2)
    return dcg / idcg


def map_at_k(r, k, fm):
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(r):
        if p == 1:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(fm, k)


def simple_test_one_user(x):
    score = x[0]
    u = x[1]

    test_pois = list(POIs - set(user_pos_train[u]))
    poi_score = []
    for i in test_pois:
        poi_score.append((i, score[i]))

    poi_score = sorted(poi_score, key=lambda x: x[1], reverse=True)
    poi_sort = [x[0] for x in poi_score]

    r = []
    for i in poi_sort:
        if i in user_pos_test[u]:
            r.append(1)  # 1 is the right prediction
        else:
            r.append(0)

    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    p_20 = np.mean(r[:20])
    p_50 = np.mean(r[:50])
    fm = float(len(user_pos_test[u]))
    r_5 = np.sum(r[:5]) / fm
    r_10 = np.sum(r[:10]) / fm
    r_20 = np.sum(r[:20]) / fm
    r_50 = np.sum(r[:50]) / fm

    map_5 = map_at_k(r, 5, fm)
    map_10 = map_at_k(r, 10, fm)
    map_20 = map_at_k(r, 20, fm)
    map_50 = map_at_k(r, 50, fm)

    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)
    ndcg_20 = ndcg_at_k(r, 20)
    ndcg_50 = ndcg_at_k(r, 50)

    return np.array(
        [p_5, p_10, p_20, p_50, r_5, r_10, r_20, r_50, map_5, map_10, map_20, map_50, ndcg_5, ndcg_10, ndcg_20,
         ndcg_50])


def simple_test(sess, model):
    result = np.array([0.] * 16)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    test_users = list(user_pos_test.keys())
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size
        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)  # ([1, 2, 3, 4], 0), ([1, 3, 4, 5], 0), ...]
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    result = result / test_user_num
    result = list(result)
    return result


def main():
    np.random.seed(8)
    param = None

    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.8)

    discriminator = Discriminator(POI_number,
                                  User_number,
                                  Embedding_size,
                                  lamda=0.1,
                                  param=param,
                                  initdelta=1,
                                  learning_rate=learning_rate)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()
    sess.run(init)
    p_p_M = get_poi_by_poi_Matrix()

    print "dis ", simple_test(sess, discriminator)
    best_p5 = 0.
    best_p10 = 0.
    best_p20 = 0.
    best_p50 = 0.
    best_r5 = 0.
    best_r10 = 0.
    best_r20 = 0.
    best_r50 = 0.
    best_m5 = 0.
    best_m10 = 0.
    best_m20 = 0.
    best_m50 = 0.
    best_d5 = 0.
    best_d10 = 0.
    best_d20 = 0.
    best_d50 = 0.

    loss_list_limit = [100] * 2   # record the historical loss

    for epoch in range(300):
        loss_per_epoch = []
        generate_dns(sess, discriminator, NEG_SAMPLE_FILE, p_p_M)  # dynamic negative sample
        with open(NEG_SAMPLE_FILE)as fin:
            for line in fin:
                line = line.split()
                u = int(line[0])
                i = int(line[1])
                j = int(line[2])
                _, my_loss, my_lr = sess.run([discriminator.D_train_step, discriminator.loss, learning_rate],
                                             feed_dict={discriminator.u: [u],
                                                        discriminator.pos: [i],
                                                        discriminator.neg: [j]})
                loss_per_epoch.append(my_loss)

        if max(loss_list_limit) < np.mean(loss_per_epoch):
            sess.run(learning_rate_decay_op)

        loss_list_limit[epoch % 2] = np.mean(loss_per_epoch)

        result = simple_test(sess, discriminator)
        print "epoch ", epoch, "dis: ", result, "loss: ", np.mean(loss_per_epoch), "lr: ", my_lr

        if result[0] > best_p5:
            best_p5 = result[0]
        if result[1] > best_p10:
            best_p10 = result[1]
        if result[2] > best_p20:
            best_p20 = result[2]
        if result[3] > best_p50:
            best_p50 = result[3]
        if result[4] > best_r5:
            best_r5 = result[4]
        if result[5] > best_r10:
            best_r10 = result[5]
        if result[6] > best_r20:
            best_r20 = result[6]
        if result[7] > best_r50:
            best_r50 = result[7]
        if result[8] > best_m5:
            best_m5 = result[8]
        if result[9] > best_m10:
            best_m10 = result[9]
        if result[10] > best_m20:
            best_m20 = result[10]
        if result[11] > best_m50:
            best_m50 = result[11]
        if result[12] > best_d5:
            best_d5 = result[12]
        if result[13] > best_d10:
            best_d10 = result[13]
        if result[14] > best_d20:
            best_d20 = result[14]
        if result[15] > best_d50:
            best_d50 = result[15]

        print "best P@5: ", best_p5
        print "best P@10: ", best_p10
        print "best P@20: ", best_p20
        print "best P@50: ", best_p50
        print "best R@5: ", best_r5
        print "best R@10: ", best_r10
        print "best R@20: ", best_r20
        print "best R@50: ", best_r50
        print "best M@5: ", best_m5
        print "best M@10: ", best_m10
        print "best M@20: ", best_m20
        print "best M@50: ", best_m50
        print "best D@5: ", best_d5
        print "best D@10: ", best_d10
        print "best D@20: ", best_d20
        print "best D@50: ", best_d50


if __name__ == '__main__':
    main()
