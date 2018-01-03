import tensorflow as tf
from data_utils import prepare_custom_data, EOS_ID, initialize_vocabulary, sentence_to_token_ids
import numpy as np
from time import time
from os import path
from math import exp
from sys import stdout, stdin
from seq2seq_model import Seq2SeqModel
class Bot:
    
    def __init__(self):
        self.buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

    def return_lines(self, _s, _t):
        return _s.readline(), _t.readline()

    def source_target_split_lens(self, source, target, EOS_ID):
        src_ids = [int(i) for i in source.split()]
        t_ids = [int(i) for i in target.split()] + [EOS_ID]
        return src_ids, t_ids, src_ids.__len__(), t_ids.__len__()

    def get_data(self, src, t_src, _max=None):
        data = [[]] * len(self.buckets)
        with tf.gfile.GFile(src, mode="r") as _src:
            with tf.gfile.GFile(t_src, mode="r") as _t:
                source, target = self.return_lines(_src, _t)
                count = 0
                while source and target and (not _max or count < _max):
                    count += 1
                    if not count % 100000:
                        stdout.flush()
                    src_ids, t_ids, len_src_ids, len_t_ids = self.source_target_split_lens(source, target, EOS_ID)
                    for _id, (src_size, t_size) in enumerate(self.buckets):
                        if len_src_ids < src_size and len_t_ids < t_size:
                            data[_id].append([src_ids, t_ids])
                            break
                    source, target = self.return_lines(_src, _t)
        return data

    def get_or_create_model(self, session, fw):
        model = Seq2SeqModel(20000,20000,self.buckets,256,3,5.0,64,0.5,0.99, forward_only=fw)
        cp = tf.train.get_checkpoint_state('cps/')
        model.saver.restore(session, cp.model_checkpoint_path) if cp else session.run(tf.initialize_all_variables())
        return model
    
    
    def get_params(self, dev_enc, dev_dec, train_enc, train_dec):
        dev_set = self.get_data(dev_enc, dev_dec)
        train_set = self.get_data(train_enc, train_dec, 0)
        train_bucket_sizes = [train_set[i].__len__() for i in xrange(self.buckets.__len__())]
        train_total_size = sum(train_bucket_sizes) * 1.0
        train_buckets_scale = [sum(train_bucket_sizes[:i+1]) / (train_total_size*1.0) for i in xrange(train_bucket_sizes.__len__())]
        return dev_set, train_set, train_buckets_scale, 0.0, 0.0, 0, []

    def train(self):
        train_enc, train_dec, dev_enc, dev_dec, _, _ = prepare_custom_data('cps/',
                'data/train.enc', 'data/train.dec', 'data/test.enc', 'data/test.dec', 20000, 20000)
        conf = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.666))
        conf.gpu_options.allocator_type = 'BFC'
        with tf.Session(config=conf) as session:
            model = self.get_or_create_model(session, False)
            dev_set,train_set,train_buckets_scale,step_time,loss,current_step,previous_losses = self.get_params(dev_enc,
                    dev_dec, train_enc, train_dec)
            while True:
                b_id = min([i for i in xrange(train_buckets_scale.__len__()) if train_buckets_scale[i] > np.random.random_sample()])
                start = time()
                encoder_inps, decoder_inps, target_weights = model.get_batch(train_set, b_id)
                _, step_loss, _ = model.step(session, encoder_inps, decoder_inps, target_weights, b_id, False)
                step_time = (time() - start)/300.0
                loss += step_loss / 300.0
                current_step += 1
                if not current_step % 300:
                    print("global step {} learning rate {} step_time {} preplexity {}".format(model.global_step.eval(), model.learning_rate.eval(), step_time, 
                        exp(loss) if loss < 300 else float('inf')))
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        session.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)
                    model.saver.save(session,path.join('cps/', 'seq2seq.ckpt'), global_step=model.global_step)
                    step_time, loss = 0.0, 0.0
                    for _b_id in xrange(self.buckets.__len__()):
                        if not dev_set[_b_id]:
                            continue
                        encoder_inps, decoder_inps, target_weights = model.get_batch(dev_set, _b_id)
                        _,eval_loss,_ = model.step(session, encoder_inps, decoder_inps, target_weights, _b_id, True)
                        print("ev: buck " + str(_b_id) + " perp "  + str(exp(eval_loss)) if eval_loss < 300 else float('inf'))
                    stdout.flush()
    
    def give_answer(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))) as session:
            model = self.get_or_create_model(session, True)
            model.batch_size = 1
            enc_vocab,_ = initialize_vocabulary(path.join('cps/','vocab20000.enc'))
            _,rev_dev_vocab = initialize_vocabulary(path.join('cps/', 'vocab20000.dec'))
            stdout.write("you: ")
            _str = stdin.readline()
            while _str:
                tok_ids = sentence_to_token_ids(tf.compat.as_bytes(_str), enc_vocab)
                b_id = min([i for i in xrange(self.buckets.__len__()) if self.buckets[0] > tok_ids.__len__()])
                encoder_inps,decoder_inps,target_weights = model.get_batch({b_id: [(tok_ids, [])]}, b_id)
                _,_,outs = model.step(session, encoder_inps, decoder_inps, target_weights, b_id, True)
                outs = [int(np.argmax(i, axis=1)) for i in  outs]
                if EOS_ID in outs:
                    outs = outs[:outs.index(EOS_ID)]
                print(" ".join(tf.compat.as_str(rev_dev_vocab[i]) for i in outs))
                stdout.write("you: ")
                stdout.flush()
                _str = stdin.readline()
