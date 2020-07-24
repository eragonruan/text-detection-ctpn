import logging
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
logger = logging.getLogger("Ealiy Stop")

class EarlyStop():
    ZERO=0
    BEST = 1
    CONTINUE = 2
    LEARNING_RATE_DECAY = 3
    STOP = -1

    def __init__(self,max_retry,max_learning_rate_decay):
        self.best_f1_value = 0
        self.learning_rate_counter = 0
        self.retry_counter = 0
        self.max_retry= max_retry
        self.max_learning_rate_decay = max_learning_rate_decay

    def decide(self,f1_value):

        if f1_value ==0:
            return EarlyStop.ZERO

        if f1_value>= self.best_f1_value:
            logger.debug("[早停]新F1值%f>旧F1值%f，记录最好的F1，继续训练",f1_value,self.best_f1_value)
            # 所有的都重置
            self.retry_counter = 0
            self.learning_rate_counter = 0
            self.best_f1_value = f1_value
            return EarlyStop.BEST

        # 甭管怎样，先把计数器++
        self.retry_counter+=1
        logger.debug("[早停]新F1值%f<旧F1值%f,早停计数器:%d", f1_value, self.best_f1_value,self.retry_counter)

        # 如果还没有到达最大尝试次数，那就继续
        if self.retry_counter < self.max_retry:
            logger.debug("[早停]早停计数器%d未达到最大尝试次数%d，继续训练",self.retry_counter,self.max_retry)
            return EarlyStop.CONTINUE

        self.learning_rate_counter+=1
        logger.debug("[早停]早停计数器大于最大尝试次数%d，学习率Decay计数器现在是:%d", self.max_retry,self.learning_rate_counter)

        # 如果还没有到达最大尝试次数，那就继续
        if self.learning_rate_counter < self.max_learning_rate_decay:
            self.retry_counter = 0 # 需要重置一下retry计数器
            logger.debug("[早停]学习率Decay计数器现在是:%d，未达到最大值%d,重置早停计数器，继续训练",self.learning_rate_counter, self.max_learning_rate_decay)
            return EarlyStop.LEARNING_RATE_DECAY

        logger.debug("[早停]学习率Decay计数器%d、早停计数器%d都已经达到最大，退出训练", self.retry_counter,self.learning_rate_counter)
        # 如果到达最大尝试次数，并且也到达了最大decay次数
        return EarlyStop.STOP