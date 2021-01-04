import numpy as np
import json
import random
import pdb
from math import ceil
from tqdm import tqdm

class Ontology:

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def add_raw(self, fname, mutual_slot):
        '''Read from json'''
        with open(fname, 'r', encoding='utf8') as f:
            self.ontology_dict = json.load(f)
        self.ontology_list = []
        for s in self.ontology_dict:
            self.ontology_list.append([s, self.ontology_dict[s]])
        self.mutual_slot = mutual_slot
        self._count()
        self._get_descartes()
    
    def add_examples(self, fname):
        with open(fname, 'r', encoding='utf8') as f:
            self.examples = json.load(f)

    def _get_descartes(self):
        self.descartes = []
        for slot, values in self.ontology_list:
            if slot == self.mutual_slot:
                continue
            for value in values:
                for status in self.ontology_dict[self.mutual_slot]:
                    item = '{}:{}-{}:{}'.format(slot, value, self.mutual_slot, status)
                    self.descartes.append(item)

    def _count(self):

        n_status = len(self.ontology_dict[self.mutual_slot])
        self.num = {
            'global': sum([len(item[1]) for item in self.ontology_list \
                if item[0] != self.mutual_slot]) * n_status
        }
        for item in self.ontology_list:
            slot, values = item
            if slot == self.mutual_slot:
                continue
            self.num[slot] = len(values) * n_status

    def add(self, fname):

        with open(fname, 'r', encoding='utf8') as f:
            ontology = json.load(f)
        self.ontology_list = ontology['ontology_list']
        self.ontology_dict = ontology['ontology_dict']
        self.mutual_slot = ontology['mutual_slot']
        self.examples = ontology['examples']
        self._count()
        self._get_descartes()

    def save(self, fname):

        with open(fname, 'w', encoding='utf8') as f:
            ontology = {
                'ontology_list': self.ontology_list,
                'ontology_dict': self.ontology_dict,
                'mutual_slot': self.mutual_slot,
                'examples': self.examples
            }
            json.dump(ontology, f, indent=4, ensure_ascii=False)

    def _get_offsets(self):
        if getattr(self, 'offsets', None) is None:
            offsets = []
            for i, item in enumerate(self.ontology_list):
                slot, values = item
                if slot == self.mutual_slot:
                    continue
                offsets.append(len(values))
                if i > 0:
                    offsets[i] += offsets[i - 1]
            self.offsets = offsets
            self.offsets = [0] + self.offsets[:-1]
    
    def _pad(self, labels_seqs):
        # 如果每种编号有n个，则第n个为<start>，n+1为<end>，n+2为<pad>
        labels_seqs_len = dict()
        for slot in labels_seqs.keys():
            pad_id = self.num[slot] + 2
            n = len(labels_seqs[slot])
            labels_seqs_len[slot] = n
            labels_seqs[slot] = labels_seqs[slot] + \
                [pad_id] * (self.num[slot] - n)

        return labels_seqs, labels_seqs_len

    def _window_labels2indexes(self, texts):

        n_status = len(self.ontology_dict[self.mutual_slot])
        labels_indexes_dict = dict()
        for slot in self.ontology_dict:
            if slot == self.mutual_slot:
                continue
            labels_indexes_dict[slot] = [0] * n_status * len(self.ontology_dict[slot])

        for text in texts:
            # 疾病:心律不齐-状态:阳性
            sv1, sv2 = text.split('-')
            s1, v1 = sv1.split(':')
            s2, v2 = sv2.split(':')
            v1_idx = self.ontology_dict[s1].index(v1)
            v2_idx = self.ontology_dict[s2].index(v2)
            slot_idx = v1_idx * n_status + v2_idx
            labels_indexes_dict[s1][slot_idx] = 1

        labels_indexes = []
        for slot, values in self.ontology_list:
            if slot == self.mutual_slot:
                continue
            labels_indexes += labels_indexes_dict[slot]

        return labels_indexes


    def labels2indexes(self, texts, style):

        self._get_offsets()
        return self._window_labels2indexes(texts)
            
    def vec2label(self, label_v):
        label_w = []
        for i, item in enumerate(label_v):
            if item == 1:
                label_w.append(self.descartes[i])
        return label_w

    def onto2ids(self):
        # max_len = max([max([len(slot) + len(value) + 1 for value in self.ontology_dict]) \
        #     for slot in self.ontology_dict])
        max_len = 35
        slots_values = dict()
        slots_values_lens = dict()
        for slot in self.ontology_dict.keys():
            for value in self.ontology_dict[slot]:
                seq = slot + ':' + value
                if len(self.examples[slot][value]) != 0:
                    seq += '('
                    for example in self.examples[slot][value]:
                        seq += example + ','
                    seq += ')'
                length = len(seq)
                seq = self.dictionary.words2ids(seq, max_len)
                try:
                    slots_values[slot].append(seq)
                    slots_values_lens[slot].append(length)
                except:
                    slots_values[slot] = [seq]
                    slots_values_lens[slot] = [length]
            slots_values[slot] = np.array(slots_values[slot], np.int32)
            slots_values_lens[slot] = np.array(slots_values_lens[slot], np.int32)
        
        return slots_values, slots_values_lens

class Corp:

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        with open(self.fname, 'r', encoding='utf8') as f:
            for line in f:
                if len(line) > 10:
                    yield list(line)

class Dictionary:

    def __init__(self):
        pass

    def load(self, fname):
        '''Load the dictionary'''
        import pdb
        with open(fname, 'r', encoding='utf8') as f:
            first_line = next(f)
            self.vocab_size = int(first_line.split(' ')[0])
            self.emb_size = int(first_line.split(' ')[1])
            self.emb = []
            self.vocab_list = []
            for line in tqdm(f):
                line_list = line.split(' ')
                if line[0] == ' ':
                    word = ' '
                    emb = np.array(list(map(lambda x: float(x), line_list[2:-1])))
                else:
                    word = line_list[0]
                    emb = np.array(list(map(lambda x: float(x), line_list[1:-1])))
                self.vocab_list.append(word)
                self.emb.append(emb)
        self.emb = np.array(self.emb, np.float32)
        self._create_index()

    def _create_index(self):

        self._id2word = self.vocab_list
        self._word2id = dict([(word,i) for i,word in enumerate(self._id2word)])

    def train(self, fname, emb_size=300):
        '''Train the word embedding'''
        self.emb_size = emb_size
        counter = Counter()
        for sentence in Corp(fname):
            counter.update(sentence)

        # 去除出现数量小于10的字
        sorted_vocab = sorted(counter.items(), key=itemgetter(1), reverse=True)
        sorted_vocab = filter(lambda x: x[1] >= 10, sorted_vocab)
        # 创建 vocab_list
        self.vocab_list = [x[0] for x in sorted_vocab]
        self.vocab_list.append('<pad>')
        self.vocab_list.append('<unk>')
        self.vocab_size = len(self.vocab_list)
        self._create_index()

        # 训练词向量
        model = Word2Vec(Corp(fname), size=emb_size, sg=1, min_count=1)
        self.emb = []
        for word in self.vocab_list[:-2]:
            self.emb.append(model[word])
        self.emb = np.array(self.emb, np.float32)
        self.emb = np.concatenate([self.emb, np.zeros([1, emb_size])], 0)
        self.emb = np.concatenate([self.emb, np.random.rand(1, emb_size) / 2], 0)
        self.emb = self.emb.astype(np.float32)

    def save(self, fname):
        with open(fname, 'w', encoding='utf8') as f:
            f.write('{} {}\n'.format(self.vocab_size, self.emb_size))
            for i, word in enumerate(self.vocab_list):
                if word == '\n':
                    word = '\\n'
                emb = list(map(lambda x: str(x), self.emb[i].tolist()))
                emb = ' '.join(emb)
                content = word + ' ' + emb + ' \n'
                f.write(content)

    def _build_norm(self):
        if getattr(self, 'emb_norm', None) is None:
            self.emb_norm = np.linalg.norm(self.emb, axis=1) # [vocab_size]

    def id2word(self, idx):
        return self._id2word[idx]

    def word2id(self, word, return_unk=False):
        try:
            return self._word2id[word]
        except KeyError:
            if return_unk:
                return self._word2id['<unk>']
            else:
                raise

    def id2emb(self, idx):
        return self.emb[idx]

    def word2emb(self, word, return_unk=False):
        idx = self.word2id(word, return_unk)
        return self.emb[idx]

    def words2ids(self, words, max_len=50):
        n = len(words)
        ids = [self.word2id(word, True) for word in words]
        if n < max_len:
            ids += [self.word2id('<pad>')] * (max_len - n)
        else:
            ids = ids[: max_len]
        return ids

    def most_similar(self, word, topn=10):
        # 获得词向量的L2范数
        self._build_norm()

        # 计算consine
        idx = self.word2id(word, False)
        emb = self.emb[idx]
        emb_norm = self.emb_norm[idx]
        dot = np.dot(self.emb, emb)
        cosine = dot / (emb_norm * self.emb_norm)
        cosine = -cosine

        # 前topn+1个数的序号数组，未排序
        most_extreme = np.argpartition(cosine, topn+1)[: topn+1]

        # 前topn+1个数排序后的序号数组
        sorted_idx = most_extreme.take(np.argsort(cosine.take(most_extreme)))

        similar_list = [(self.id2word(idx), -cosine[idx]) for idx in sorted_idx[1:]]
        return similar_list

class Data:
    def __init__(self, max_len, dictionary, ontology):
        self.max_len = max_len
        self.datasets = dict()
        self.dictionary = dictionary
        self.ontology = ontology

    def _add_window(self, name, fname):
        with open(fname, 'r', encoding='utf8') as f:
            dialogs = json.load(f)
        self.datasets[name] = {
            'origin': dialogs,
            'windows_utts': [],
            'windows_utts_len': [],
            'labels': []
        }
        for dialog in dialogs:
            for window in dialog:
                window_utts = window['utterances']
                self.window_size = len(window_utts)
                window_utts_len = [len(utt) for utt in window_utts]
                self.datasets[name]['windows_utts_len'].append(window_utts_len)
                window_utts = [self.dictionary.words2ids(utt, self.max_len) \
                    for utt in window_utts]
                self.datasets[name]['windows_utts'].append(window_utts)
                label = window['label']
                label = filter(lambda x: '药物' not in x, label)
                label_indexes = self.ontology.labels2indexes(label, 'window')
                self.datasets[name]['labels'].append(label_indexes)
        self.datasets[name]['windows_utts'] = np.array(
            self.datasets[name]['windows_utts'], np.int32)
        self.datasets[name]['windows_utts_len'] = np.array(
            self.datasets[name]['windows_utts_len'], np.int32)
        self.datasets[name]['labels'] = np.array(
            self.datasets[name]['labels'], np.int32)
        self.datasets[name]['style'] = 'window'
        self.datasets[name]['num'] = self.datasets[name]['windows_utts'].shape[0]

    def add_raw(self, name, fname, style):
        if style == 'window':
            self._add_window(name, fname)
        elif style == 'window-chunk':
            self._add_window_chunk(name, fname)

    def _add_window_chunk(self, name, fname):
        with open(fname, 'r', encoding='utf8') as f:
            dialogs = json.load(f)
        self.datasets[name] = {
            'origin': dialogs,
            'chunk_seqs': [],
            'chunk_seqs_len': [],
            'labels': []
        }
        for dialog in dialogs:
            for window in dialog:
                utts = window['utterances']
                chunk_seq = []
                for utt in utts:
                    if len(utt) != 0:
                        chunk_seq += list(utt) + ['<seg>']
                    else:
                        chunk_seq += list(utt)
                chunk_seq = chunk_seq[:-1]
                chunk_seq_len = len(chunk_seq)
                chunk_seq_ids = self.dictionary.words2ids(chunk_seq, self.max_len)

                self.datasets[name]['chunk_seqs'].append(chunk_seq_ids)
                self.datasets[name]['chunk_seqs_len'].append(chunk_seq_len)

                label = window['label']
                label = filter(lambda x: '药物' not in x, label)
                label_indexes = self.ontology.labels2indexes(label, 'window')
                self.datasets[name]['labels'].append(label_indexes)

        self.datasets[name]['chunk_seqs'] = np.array(
            self.datasets[name]['chunk_seqs'], np.int32)
        self.datasets[name]['chunk_seqs_len'] = np.array(
            self.datasets[name]['chunk_seqs_len'], np.int32)
        self.datasets[name]['labels'] = np.array(
            self.datasets[name]['labels'], np.int32)
        self.datasets[name]['style'] = 'window-chunk'
        self.datasets[name]['num'] = self.datasets[name]['chunk_seqs'].shape[0]

    def _batch_window(self, name, batch_size, shuffle=True):
        n = self.datasets[name]['num']
        batch_num = int(ceil(n / batch_size))
        indexes = list(range(n))
        if shuffle:
            random.shuffle(indexes)
        for i in range(batch_num):
            indexes_batch = indexes[i * batch_size: (i + 1) * batch_size]
            windows_utts_batch = self.datasets[name]['windows_utts'].\
                take(indexes_batch, 0)
            windows_utts_lens_batch = self.datasets[name]['windows_utts_len'].\
                take(indexes_batch, 0)
            labels_batch = self.datasets[name]['labels'].\
                take(indexes_batch, 0)
            yield windows_utts_batch, windows_utts_lens_batch, labels_batch

    def _batch_window_chunk(self, name, batch_size, shuffle=True):
        n = self.datasets[name]['num']
        batch_num = int(ceil(n / batch_size))
        indexes = list(range(n))
        if shuffle:
            random.shuffle(indexes)
        for i in range(batch_num):
            indexes_batch = indexes[i * batch_size: (i + 1) * batch_size]
            chunk_seqs_batch = self.datasets[name]['chunk_seqs'].\
                take(indexes_batch, 0)
            chunk_seqs_len_batch = self.datasets[name]['chunk_seqs_len'].\
                take(indexes_batch, 0)
            labels_batch = self.datasets[name]['labels'].\
                take(indexes_batch, 0)
            chunk_seqs_batch = chunk_seqs_batch[:, None, :]
            chunk_seqs_len_batch = chunk_seqs_len_batch[:, None]
            yield chunk_seqs_batch, chunk_seqs_len_batch, labels_batch

    def batch(self, name, batch_size, shuffle=True):
        if self.datasets[name]['style'] == 'window':
            yield from self._batch_window(name, batch_size, shuffle)
        elif self.datasets[name]['style'] == 'window-chunk':
            yield from self._batch_window_chunk(name, batch_size, shuffle)