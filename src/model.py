import tensorflow as tf
import numpy as np
import random
import pinyin
import os
import json

from tqdm import tqdm
from math import ceil

INF = 1e5
DEFAULT_BATCH_SIZE = 1000
MAX_WINDOW_SIZE = 200

class MIE:

    def __init__(self, data, ontology, **kw):
        # 初始化data，ontology
        self.data = data
        self.ontology = ontology
        self.slots = [item[0] for item in self.ontology.ontology_list \
            if item[0] != self.ontology.mutual_slot]
        # self.weights = [len(values) for _, values in self.ontology.ontology_list]

        self.max_len = self.data.max_len
        # 创建输入和输出的占位符
        self.windows_utts = tf.placeholder(tf.int32, [None, None, self.max_len])
        # [batch_size, window_size, max_len]
        self.windows_utts_lens = tf.placeholder(tf.int32, [None, None])
        # [batch_size, window_size]

        self.labels = tf.placeholder(tf.float32, [None, len(self.ontology.descartes)])
        # [batch_size, slots_values_num]
        self.keep_p = tf.placeholder(tf.float32, [])
        self.lr = tf.placeholder(tf.float32, [])
        self.window_size = tf.placeholder(tf.int32, [])
        self.batch_size = tf.placeholder(tf.int32, [])

        # 如果关键字参数是location，则导入模型参数，建立计算图
        if 'location' in kw:
            self._load(kw['location'])
        # 如果关键字参数是paras，则根据参数建立计算图
        elif 'params' in kw:
            self.params = kw['params']
            self._build()
            self.reinit()

    def reinit(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.infos = dict()
        for dataset in ('train', 'dev'):
            self.infos[dataset] = dict()
            for slot in self.slots:
                self.infos[dataset][slot] = {
                    'ps': [],
                    'rs': [],
                    'f1s' : [],
                    'losses': []
                }
            self.infos[dataset]['global'] = {
                'ps': [],
                'rs': [],
                'f1s' : [],
                'losses': []
            }

    def _bilstm(self, slot, seqs, seqs_lens):
        num_units = self.params['num_units']
        with tf.variable_scope(slot, reuse=tf.AUTO_REUSE):
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.contrib.rnn.LSTMCell(num_units / 2),
                input_keep_prob=self.keep_p,
                output_keep_prob=self.keep_p
            )
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.contrib.rnn.LSTMCell(num_units / 2),
                input_keep_prob=self.keep_p,
                output_keep_prob=self.keep_p
            )
            hidden_states, last_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, 
                cell_bw=cell_bw,
                inputs=seqs,
                sequence_length=seqs_lens,
                dtype=tf.float32
            )
            hidden_states = tf.concat(hidden_states, -1)
            # [batch_size, max_len, num_units]
        return hidden_states

    def _gate(self, slot, input1, input2):
        with tf.variable_scope(slot, reuse=tf.AUTO_REUSE):
            beta = tf.get_variable('beta', initializer=tf.constant(0.5))
            output = beta * input1 + (1 - beta) * input2
        return output

    def _self_atten(self, slot, seqs):
        with tf.variable_scope(slot, reuse=tf.AUTO_REUSE):
            a = tf.layers.dense(seqs, 1)
            # [num, max_len, 1]
            point = tf.expand_dims(tf.reduce_sum(seqs, -1), -1)
            # [num, max_len, 1]
            mask = -tf.cast(tf.equal(point, 0), tf.float32) * INF
            a = a + mask
            p = tf.nn.softmax(a, 1)
            # [num, max_len, 1]
            c = tf.reduce_sum(p * seqs, 1)
            # [num, emb_size]
            c = tf.nn.dropout(c, self.keep_p)
        return c

    def _encoder(self, encoder_name, slot, seqs, seqs_lens):
        with tf.variable_scope(encoder_name, reuse=tf.AUTO_REUSE):
            if self.params['add_global']:
                h_s = self._bilstm(slot, seqs, seqs_lens)
                h_g = self._bilstm('global', seqs, seqs_lens)
                h = self._gate(slot + '-1st', h_s, h_g)
                c_s = self._self_atten(slot, h)
                c_g = self._self_atten('global', h)
                c = self._gate(slot + '-2nd', c_s, c_g)
            else:
                h = self._bilstm(slot, seqs, seqs_lens)
                c = self._self_atten(slot, h)
            
        return h, c

    def _position_encoding(self):
        num_units = self.params['num_units']
        sin = lambda pos, i: np.sin(pos/(1000**(i / num_units))) # i 是偶数
        cos = lambda pos, i: np.cos(pos/(1000**((i-1) / num_units))) # i 是奇数
        PE = [[sin(pos,i) if i%2==0 else cos(pos,i) for i in range(num_units)]\
            for pos in range(MAX_WINDOW_SIZE)]
        PE = tf.constant(np.array(PE), dtype=tf.float32) # [MAX_WINDOW_SIZE, num_units]
        return PE

    def _attention(
            self,
            query, # [slot_value_num, num_units]
            keys, # [batch_size, window_size, max_len, num_units]
            values # [batch_size, window_size, max_len, num_units]
        ):
        slot_value_num = query.shape[0]

        query = tf.tile(
            tf.expand_dims(tf.expand_dims(query, 0), 0),
            [self.batch_size, self.window_size, 1, 1]
        ) # [batch_size, window_size, slot_value_num, num_units]
        p = tf.matmul(
            query,
            tf.transpose(keys, [0, 1, 3, 2]) # [batch_size, window_size, num_units, max_len]
        ) # [batch_size, window_size, slot_value_num, max_len]

        mask = - tf.cast(tf.equal(p, 0), tf.float32) * INF
        p = tf.nn.softmax(p + mask, -1)


        outputs = tf.matmul(p, values)
        # [batch_size, window_size, slot_value_num, num_units]

        return outputs

    def _feedforward(self,
                    slot,
                    inputs,
                    num_layers,
                    num_units,
                    outputs_dim,
                    activation,
                    keep_p):
        with tf.variable_scope(slot):
            if num_layers != 0:
                outputs = tf.layers.dense(
                    inputs,
                    num_units,
                    activation=activation
                )
                outputs = tf.nn.dropout(outputs, keep_p)
                for i in range(num_layers - 1):
                    outputs = tf.layers.dense(
                        outputs,
                        num_units,
                        activation=activation
                    )
                    outputs = tf.nn.dropout(outputs, keep_p)
            else:
                outputs = inputs
            outputs = tf.layers.dense(outputs, outputs_dim)
            return outputs

    def _build(self):
        raise NotImplementedError()

    def _create_sess(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()

    def inference(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        if num < 0:
            num = INF
        slots_pred_labels = [[] for i in range(len(self.slots_pred_labels))]
        slots_gold_labels = [[] for i in range(len(self.slots_pred_labels))]
        for i, batch in enumerate(self.data.batch(name, batch_size, False)):
            if (i + 1) * batch_size > num:
                break
            windows_utts_batch, windows_utts_lens_batch, labels_batch = batch
            true_batch_size = windows_utts_batch.shape[0]
            window_size = windows_utts_batch.shape[1]
            slots_pred_labels_batch = self.sess.run(self.slots_pred_labels, feed_dict={
                self.windows_utts: windows_utts_batch,
                self.windows_utts_lens: windows_utts_lens_batch,
                self.window_size: window_size,
                self.batch_size: true_batch_size,
                self.keep_p: 1
            }) # [batch_size, n * num_statues] * num_slots
            start = 0
            for i, slot_pred_labels_batch in enumerate(slots_pred_labels_batch):
                end = start + slot_pred_labels_batch.shape[1]
                slots_gold_labels[i].append(labels_batch[:, start: end])
                slots_pred_labels[i].append(slot_pred_labels_batch)
                start = end
        # slots_pred_labels为一个num_slots个元素的列表，每个元素为[num, n * num_statues]
        for i in range(len(slots_gold_labels)):
            slots_gold_labels[i] = np.concatenate(slots_gold_labels[i], 0)
            slots_pred_labels[i] = np.concatenate(slots_pred_labels[i], 0)

        return slots_pred_labels, slots_gold_labels

    def compute_loss(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        if num < 0:
            num = INF
        slots_loss = [[] for i in range(len(self.slots_loss))]
        for i, batch in enumerate(self.data.batch(name, batch_size, False)):
            if (i + 1) * batch_size > num:
                break
            windows_utts_batch, windows_utts_lens_batch, labels_batch = batch
            true_batch_size = windows_utts_batch.shape[0]
            window_size = windows_utts_batch.shape[1]
            slots_loss_batch = self.sess.run(self.slots_loss, feed_dict={
                self.windows_utts: windows_utts_batch,
                self.windows_utts_lens: windows_utts_lens_batch,
                self.window_size: window_size,
                self.batch_size: true_batch_size,
                self.labels: labels_batch,
                self.keep_p: 1
            }) # [batch_size, n * num_statues] * num_slots
            for i, slot_loss_batch in enumerate(slots_loss_batch):
                slots_loss[i].append(slot_loss_batch)

        for i in range(len(slots_loss)):
            slots_loss[i] = np.concatenate(slots_loss[i], 0)

        losses = dict([(slot, None) for slot in self.slots])
        losses['global'] = None

        for i, slot_loss in enumerate(slots_loss):
            slot = self.slots[i]
            loss = float(np.mean(slot_loss))
            losses[slot] = loss

        losses['global'] = float(np.mean(np.concatenate(slots_loss, -1)))

        return losses

    def train(self,
            epoch_num,
            batch_size,
            tbatch_size,
            start_lr,
            end_lr,
            location=None):
        # 计算衰减率
        decay = (end_lr / start_lr) ** (1 / epoch_num)
        lr = start_lr

        save_graph = True
        num_slots = len(self.slots_train_op)
        indexes = list(range(num_slots))

        for i in range(epoch_num):
            random.shuffle(indexes)
            pbar = tqdm(
                self.data.batch('train', batch_size, True),
                desc='Epoch {}:'.format(i + 1),
                total=ceil(self.data.datasets['train']['num'] / batch_size)
            )
            for batch in pbar:
                windows_utts_batch, windows_utts_lens_batch, labels_batch = batch
                true_batch_size = windows_utts_batch.shape[0]
                window_size = windows_utts_batch.shape[1]
                self.sess.run(
                    [self.slots_train_op[j] for j in indexes],
                    feed_dict={
                        self.windows_utts: windows_utts_batch,
                        self.windows_utts_lens: windows_utts_lens_batch,
                        self.labels: labels_batch,
                        self.window_size: window_size,
                        self.batch_size: true_batch_size,
                        self.lr: lr,
                        self.keep_p: self.params['keep_p']
                    }
                )
            pbar.close()
            lr *= decay

            train_prf = self.evaluate('train', tbatch_size, tbatch_size)
            train_loss = self.compute_loss('train', tbatch_size, tbatch_size)
            dev_prf = self.evaluate('dev', batch_size=tbatch_size)
            dev_loss = self.compute_loss('dev', batch_size=tbatch_size)

            self._add_infos('train', train_prf)
            self._add_infos('train', train_loss)
            self._add_infos('dev', dev_prf)
            self._add_infos('dev', dev_loss)

            # 打印信息
            print('''Epoch {}: train_loss={:.4}, dev_loss={:.4}
                train_p={:.4}, train_r={:.4}, train_f1={:.4}
                dev_p={:.4}, dev_r={:.4}, dev_f1={:.4}'''.
                format(i + 1, train_loss['global'], dev_loss['global'],
                    train_prf['global']['p'], train_prf['global']['r'], 
                    train_prf['global']['f1'], dev_prf['global']['p'],
                    dev_prf['global']['r'], dev_prf['global']['f1']))

            if len(self.infos['dev']['global']['f1s']) > 0 and location:
                if dev_prf['global']['f1'] >= max(self.infos['dev']['global']['f1s']):
                    test_prf = self.evaluate('test', batch_size=tbatch_size)
                    self.save(location, save_graph)
                    save_graph = False
                    print('保存在{}！'.format(location))

            try:
                print('Now test result: f1={:.4}'.format(test_prf['global']['f1']))
            except NameError:
                pass

    def _evaluate(self, pred_labels, gold_labels):
        def _add_ex_col(x):
            col = 1 - np.sum(x, -1).astype(np.bool).astype(np.float32)
            col = np.expand_dims(col, -1)
            x = np.concatenate([x, col], -1)
            return x
        pred_labels = _add_ex_col(pred_labels)
        gold_labels = _add_ex_col(gold_labels)
        tp = np.sum((pred_labels == gold_labels).astype(np.float32) * pred_labels, -1)
        pred_pos_num = np.sum(pred_labels, -1)
        gold_pos_num = np.sum(gold_labels, -1)
        p = (tp / pred_pos_num)
        r = (tp / gold_pos_num)
        p_add_r = p + r
        p_add_r = p_add_r + (p_add_r == 0).astype(np.float32)
        f1 = 2 * p * r / p_add_r

        return p, r, f1

    def evaluate(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        slots_pred_labels, slots_gold_labels = \
            self.inference(name, num, batch_size)

        info = dict()
        for slot in self.slots:
            info[slot] = {
                'p': None,
                'r': None,
                'f1': None
            }
        info['global'] = {
            'p': None,
            'r': None,
            'f1': None
        }

        for i, (slot_pred_labels, slot_gold_labels) in \
            enumerate(zip(slots_pred_labels, slots_gold_labels)):
            p, r, f1 = map(
                lambda x: float(np.mean(x)),
                self._evaluate(slot_pred_labels, slot_gold_labels)
            )
            slot = self.slots[i]
            info[slot]['p'] = p
            info[slot]['r'] = r
            info[slot]['f1'] = f1

        pred_labels = np.concatenate(slots_pred_labels, -1)
        gold_labels = np.concatenate(slots_gold_labels, -1)

        p, r, f1 = map(
            lambda x: float(np.mean(x)),
            self._evaluate(pred_labels, gold_labels)
        )
        info['global']['p'] = p
        info['global']['r'] = r
        info['global']['f1'] = f1

        return info

    def _add_infos(self, name, info):
        for slot in info.keys():
            if isinstance(info[slot], float):
                # 说明是loss
                self.infos[name][slot]['losses'].append(info[slot])
            elif isinstance(info[slot], dict):
                # 说明是p r f
                for key in info[slot].keys():
                    self.infos[name][slot][key + 's'].append(info[slot][key])


    def _build(self):
        '''Build the computational graph'''
        windows_utts = tf.nn.embedding_lookup(
            params=self.data.dictionary.emb,
            ids=self.windows_utts
        ) # [batch_size, window_size, max_len, emb_size]

        candidate_seqs_dict, candidate_seqs_lens_dict = self.ontology.onto2ids()
        for slot in candidate_seqs_dict.keys():
            candidate_seqs_dict[slot] = tf.nn.embedding_lookup(
                params=self.data.dictionary.emb,
                ids=candidate_seqs_dict[slot]
            )
        # dim = tf.reduce_prod(tf.shape(windows_utts[:2]))
        utts = tf.reshape(windows_utts, [-1, self.max_len, self.data.dictionary.emb_size])
        # [batch_size * window_size, max_len, emb_size]
        utts_lens = tf.reshape(self.windows_utts_lens, [-1])
        # [batch_size * window_size]
        slot_utt_hs_dict = dict()
        slot_candidate_cs_dict = dict()

        for slot, _ in self.ontology.ontology_list:
            utt_h, _ = self._encoder(
                'utt_encoder',
                pinyin.get(slot, format='strip'),
                utts,
                utts_lens
            )
            # [batch_size * window_size, max_len, num_units]
            utt_h = tf.reshape(utt_h, [-1, self.window_size, self.max_len, self.params['num_units']])
            # [batch_size, window_size, max_len, num_units]

            _, candidate_c = self._encoder(
                'candidate_encoder',
                pinyin.get(slot, format='strip'),
                candidate_seqs_dict[slot],
                candidate_seqs_lens_dict[slot]
            ) # [slot_value_num, num_units]

            if slot == self.ontology.mutual_slot:
                status_utt_h = utt_h
                # [batch_size, window_size, max_len, num_units]
                status_candidate_c = candidate_c
            else:
                slot_utt_hs_dict[slot] = utt_h
                slot_candidate_cs_dict[slot] = candidate_c
        
        mask = - tf.cast(
            tf.equal(tf.expand_dims(self.windows_utts_lens, -1), 0),
            tf.float32
        ) * INF
        # [batch_size, window_size, 1]

        position_encoding = self._position_encoding()
        position_encoding = tf.expand_dims(
            tf.expand_dims(
                position_encoding[:self.window_size],
                0
            ),
            2
        )

        q_status = self._attention(status_candidate_c, status_utt_h, status_utt_h) \
            + position_encoding
        # [batch_size, window_size, status_num, num_units]

        start = 0
        self.slots_pred_logits = []
        self.slots_pred_labels = []
        self.slots_gold_labels = []
        self.slots_loss = []
        self.slots_train_op = []
        for slot in self.slots:
            # slot_utt_h和slot_candidate_c
            # status_utt_h和status_candidate_c

            slot_utt_h = slot_utt_hs_dict[slot]
            # [batch_size, window_size, max_len, num_units]
            slot_candidate_c = slot_candidate_cs_dict[slot]
            # [slot_value_num, num_units]

            q_slot = self._attention(slot_candidate_c, slot_utt_h, slot_utt_h) \
                + position_encoding
            # [batch_size, window_size, slot_value_num, num_units]

            slot_value_num = slot_candidate_c.shape[0]
            status_num = status_candidate_c.shape[0]

            # q_slot和q_status进行attention

            w = tf.get_variable(
                pinyin.get(slot, format='strip') + '-w',
                [1, self.params['num_units'], self.params['num_units']],
                tf.float32
            )
            w = tf.tile(
                w,
                [self.batch_size, 1, 1]
            ) # [batch_size, num_units, num_units]
            co = tf.reshape(
                tf.reshape(
                    tf.matmul(
                        tf.matmul(
                            tf.reshape(
                                q_slot,
                                [self.batch_size, self.window_size * slot_value_num, self.params['num_units']]
                            ),
                            w
                        ),
                        tf.reshape(
                            q_status,
                            [self.batch_size, self.window_size * status_num, self.params['num_units']]
                        ),
                        transpose_b=True
                    ), # [batcn_size, window_size * slot_value_num, window_size * status_num]
                    [self.batch_size, self.window_size, slot_value_num, self.window_size * status_num]
                ),
                [self.batch_size, self.window_size, slot_value_num, self.window_size, status_num]
            )
            co_mask = - tf.cast(tf.equal(co, 0), tf.float32) * INF
            p = co + co_mask

            p = tf.nn.softmax(p, 3)
            q_status_slot = tf.transpose(
                tf.tile(
                    tf.expand_dims(tf.expand_dims(q_status, -1), -1),
                    [1, 1, 1, 1, self.window_size, slot_value_num]
                ),
                [0, 4, 5, 1, 2, 3]
            )# [batch_size, window_size, slot_value_num, window_size, status_num, num_units]
            q_status_slot = tf.reduce_sum(tf.multiply(
                tf.expand_dims(p, -1),
                q_status_slot
            ), 3) # [batch_size, window_size, slot_value_num, status_num, num_units]

            q_slot = tf.tile(tf.expand_dims(q_slot, 3), [1, 1, 1, status_num, 1])
            features = tf.concat([q_slot, q_status_slot], -1)
            # [batch_size, window_size, slot_value_num, status_num, 2 * num_units]
            logits = self._feedforward(
                pinyin.get(slot, format='strip'),
                features,
                self.params['num_layers'],
                self.params['num_units'],
                1,
                tf.nn.relu,
                self.keep_p
            ) # [batch_size, window_size, slot_value_num, status_num, 1]

            logits = tf.reshape(
                logits,
                [-1, self.window_size, slot_value_num * status_num]
            )
            # [batch_size, window_size, slot_value_num * status_num]

            slot_pred_logits = tf.reduce_max(logits + mask, 1)
            # [batch_size, slot_value_num * status_num]

            self.slots_pred_logits.append(slot_pred_logits)

            # 当前slot的输出标签
            slot_pred_labels = tf.cast(slot_pred_logits > 0, tf.float32)
            self.slots_pred_labels.append(slot_pred_labels)

            # 当前slot的真实标签
            slot_gold_labels = self.labels[:, start: start + slot_value_num * status_num]
            self.slots_gold_labels.append(slot_gold_labels)

            # 当前slot的loss
            slot_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=slot_pred_logits,
                labels=slot_gold_labels
            ) # [batch_size, status_num * slot_value_num]
            self.slots_loss.append(slot_loss)

            # 当前slot的训练节点
            slot_train_op = tf.train.AdamOptimizer(self.lr).\
                minimize(tf.reduce_mean(slot_loss))
            self.slots_train_op.append(slot_train_op)

            start += slot_value_num * status_num

        self._create_sess()

    def save(self, location, save_graph=True):
        if not os.path.exists(location):
            os.makedirs(location)
        with open(os.path.join(location, 'params.json'), 'w', encoding='utf8') as f:
            json.dump(self.params, f, indent=4, ensure_ascii=False)
        with open(os.path.join(location, 'infos.json'), 'w', encoding='utf8') as f:
            json.dump(self.infos, f, indent=4, ensure_ascii=False)
        self.saver.save(
            self.sess,
            os.path.join(location, 'model.ckpt'),
            write_meta_graph=save_graph
        )

    def _load(self, location):
        with open(os.path.join(location, 'params.json'), 'r', encoding='utf8') as f:
            self.params = json.load(f)
        with open(os.path.join(location, 'infos.json'), 'r', encoding='utf8') as f:
            self.infos = json.load(f)
        self._build()
        self.saver.restore(self.sess, os.path.join(location, 'model.ckpt'))

    def restore(self, fname):
        self.saver.restore(self.sess, fname)

    def close(self):
        self.sess.close()