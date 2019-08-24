from typing import Any, Union
from collections import namedtuple
import numbers
import heapq
import random
import math

import dynet as dy
import numpy as np

from xnmt import batchers, param_collections, expression_seqs
from xnmt.modelparts import bridges, transforms, scorers, embedders
from xnmt.transducers import recurrent
from xnmt.persistence import serializable_init, Serializable, bare, Ref
from xnmt.vocabs import Vocab, RnngVocab, RnngAction

class Decoder(object):
  """
  A template class to convert a prefix of previously generated words and
  a context vector into a probability distribution over possible next words.
  """
  def initial_state(self, enc_final_states, ss_expr):
    raise NotImplementedError('must be implemented by subclasses')
  def add_input(self, mlp_dec_state, trg_embedding):
    raise NotImplementedError('must be implemented by subclasses')

  def calc_loss(self, x, ref_action):
    raise NotImplementedError('must be implemented by subclasses')
  def calc_score(self, x, action, normalize=False):
    raise NotImplementedError('must be implemented by subclasses')

  def best_k(self, x, k=1, normalize_scores=False):
    raise NotImplementedError('must be implemented by subclasses')
  def sample(self, x, n=1, temperature=1.0):
    raise NotImplementedError('must be implemented by subclasses')

class DecoderState(object):
  """A state that holds whatever information is required for the decoder.
     Child classes must implement the as_vector() method, which will be
     used by e.g. the attention mechanism"""

  def as_vector(self):
    raise NotImplementedError('must be implemented by subclass')

  def is_complete(self):
    raise NotImplementedError('must be implemented by subclass')

class AutoRegressiveDecoderState(DecoderState):
  """A state holding all the information needed for AutoRegressiveDecoder

  Args:
    rnn_state: a DyNet RNN state
    context: a DyNet expression
    complete: boolean representing whether this state represents a complete
              sentence
  """
  def __init__(self, rnn_state: recurrent.UniLSTMState = None, context: dy.Expression = None, complete: Union[bool, batchers.ListBatch] = False):
    self.rnn_state = rnn_state
    self.context = context
    self.complete = complete

    batch_size = context.dim()[1]
    if batch_size > 1:
      if not batchers.is_batched(complete):
        self.complete = batchers.ListBatch([complete for _ in range(batch_size)])

  def as_vector(self):
    return self.rnn_state.output()

  def is_complete(self):
    if batchers.is_batched(self.complete):
      return not (False in self.complete)
    else:
      return self.complete

class AutoRegressiveDecoder(Decoder, Serializable):
  """
  Standard autoregressive-decoder.

  Args:
    input_dim: input dimension
    embedder: embedder for target words
    input_feeding: whether to activate input feeding
    bridge: how to initialize decoder state
    rnn: recurrent decoder
    transform: a layer of transformation between rnn and output scorer
    scorer: the method of scoring the output (usually softmax)
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
  """

  yaml_tag = '!AutoRegressiveDecoder'

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               embedder: embedders.Embedder = bare(embedders.SimpleWordEmbedder),
               input_feeding: bool = True,
               bridge: bridges.Bridge = bare(bridges.CopyBridge),
               rnn: recurrent.UniLSTMSeqTransducer = bare(recurrent.UniLSTMSeqTransducer),
               transform: transforms.Transform = bare(transforms.AuxNonLinear),
               scorer: scorers.Scorer = bare(scorers.Softmax),
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    self.param_col = param_collections.ParamManager.my_params(self)
    self.input_dim = input_dim
    self.embedder = embedder
    self.truncate_dec_batches = truncate_dec_batches
    self.bridge = bridge
    self.rnn = rnn
    self.transform = transform
    self.scorer = scorer
    # Input feeding
    self.input_feeding = input_feeding
    rnn_input_dim = embedder.emb_dim
    if input_feeding:
      rnn_input_dim += input_dim
    assert rnn_input_dim == rnn.total_input_dim, "Wrong input dimension in RNN layer: {} != {}".format(rnn_input_dim, rnn.total_input_dim)

  def shared_params(self):
    return [{".embedder.emb_dim", ".rnn.input_dim"},
            {".input_dim", ".rnn.decoder_input_dim"},
            {".input_dim", ".transform.input_dim"},
            {".input_feeding", ".rnn.decoder_input_feeding"},
            {".rnn.layers", ".bridge.dec_layers"},
            {".rnn.hidden_dim", ".bridge.dec_dim"},
            {".rnn.hidden_dim", ".transform.aux_input_dim"},
            {".transform.output_dim", ".scorer.input_dim"}]

  def initial_state(self, enc_final_states: Any, ss: Any) -> AutoRegressiveDecoderState:
    """Get the initial state of the decoder given the encoder final states.

    Args:
      enc_final_states: The encoder final states. Usually but not necessarily an :class:`xnmt.expression_sequence.ExpressionSequence`
      ss: first input
    Returns:
      initial decoder state
    """
    rnn_state = self.rnn.initial_state()
    rnn_s = self.bridge.decoder_init(enc_final_states)
    rnn_state = rnn_state.set_s(rnn_s)
    batch_size = rnn_s[0].dim()[1]
    zeros = dy.zeros(self.input_dim, batch_size=batch_size) if self.input_feeding else None
    ss_expr = self.embedder.embed(ss)
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros]) if self.input_feeding else ss_expr)
    return AutoRegressiveDecoderState(rnn_state=rnn_state, context=zeros)

  def add_input(self, mlp_dec_state: AutoRegressiveDecoderState, trg_word: Any) -> AutoRegressiveDecoderState:
    """Add an input and update the state.

    Args:
      mlp_dec_state: An object containing the current state.
      trg_word: The word to input.
    Returns:
      The updated decoder state.
    """
    assert not mlp_dec_state.is_complete(), 'Attempt to add to a complete hypothesis!'
    trg_embedding = self.embedder.embed(trg_word)
    inp = trg_embedding
    if self.input_feeding:
      inp = dy.concatenate([inp, mlp_dec_state.context])
    rnn_state = mlp_dec_state.rnn_state
    if self.truncate_dec_batches:
      rnn_state, inp = batchers.truncate_batches(rnn_state, inp)
    new_rnn_state = rnn_state.add_input(inp)
    try:
      complete = batchers.mark_as_batch([w == Vocab.ES for w in trg_word])
    except TypeError:
      complete = (trg_word == Vocab.ES)
    new_state =  AutoRegressiveDecoderState(
        rnn_state=new_rnn_state,
        context=mlp_dec_state.context,
        complete=complete)
    return new_state

  def _calc_transform(self, mlp_dec_state: AutoRegressiveDecoderState) -> dy.Expression:
    h = dy.concatenate([mlp_dec_state.as_vector(), mlp_dec_state.context])
    return self.transform.transform(h)

  def best_k(self, mlp_dec_state: AutoRegressiveDecoderState, k: numbers.Integral, normalize_scores: bool = False):
    h = self._calc_transform(mlp_dec_state)
    best_words, best_scores = self.scorer.best_k(h, k, normalize_scores=normalize_scores)
    return best_words, best_scores

  def sample(self, mlp_dec_state: AutoRegressiveDecoderState, n: numbers.Integral, temperature: float = 1.0):
    h = self._calc_transform(mlp_dec_state)
    return self.scorer.sample(h, n, temperature)

  def calc_log_probs(self, mlp_dec_state):
    #raise NotImplementedError('deprecated')
    return self.scorer.calc_log_probs(self._calc_transform(mlp_dec_state))

  def calc_loss(self, mlp_dec_state, ref_action):
    return self.scorer.calc_loss(self._calc_transform(mlp_dec_state), ref_action)

# Each item on a RnngDecoderState's stack is a tuple of:
# 1) an nt_id
# 2) a list of children
# 3) the output of the stack_lstm up to, but not including, this NT
RnngStackElement = namedtuple('RnngStackElement', 'nt_id, children, prev_state')

class RnngDecoderState(object):
  MaxOpenNTs = 100

  def __init__(self, stack_lstm_state=None, term_lstm_state=None, action_lstm_state=None):
    self.stack = [] # Stack of RnngStackElements
    self.terminals = [] # Generated terminals
    self.actions = [] # Generated action sequence
    self.term_lstm_state = term_lstm_state
    self.action_lstm_state = action_lstm_state
    self.stack_lstm_state = stack_lstm_state
    self.context = None

    self.pos_embs = {}

  def make_pos_emb(self, pos):
    self.hidden_dim = 512
    assert self.hidden_dim % 2 == 0
    v = [0. for _ in range(self.hidden_dim)]
    for i in range(self.hidden_dim // 2):
      v[2 * i] = math.sin(pos / 10000. ** (2 * i / self.hidden_dim))
      v[2 * i + 1] = math.cos(pos / 10000. ** (2 * i / self.hidden_dim))
    return np.array(v)

  def get_pos_emb(self, pos):
    if pos not in self.pos_embs:
      self.pos_embs[pos] = self.make_pos_emb(pos)
    return dy.inputTensor(self.pos_embs[pos])

  def __str__(self):
    return '\n'.join(['stack: ' + str(self.stack), 'terms:' + str(self.terminals)])

  def is_forbidden(self, a : RnngAction):
    if a.action not in [RnngVocab.SHIFT, RnngVocab.NT, RnngVocab.REDUCE, RnngVocab.NONE]:
      return True

    if len(self.stack) >= RnngDecoderState.MaxOpenNTs:
      if a.action == RnngVocab.NT:
        return True

    if len(self.stack) == 0:
      if a.action != RnngVocab.NT and a.action != RnngVocab.NONE:
        return True

    if a.action == RnngVocab.REDUCE:
      if len(self.stack) == 0:
        return True
      if len(self.stack[-1].children) == 0:
        return True

    if a.action == RnngVocab.NONE:
      return not self.is_complete()

    return False

  def as_vector(self):
    stack_vec = self.stack_lstm_state.output()
    r = stack_vec

    if self.term_lstm_state is not None:
      terms_vec = self.term_lstm_state.output()
      r += terms_vec

    if self.action_lstm_state is not None:
      actions_vec = self.action_lstm_state.output()
      r += actions_vec

    #r += self.get_pos_emb(len(self.stack))
    return r

  def is_complete(self):
    return len(self.terminals) > 0 and len(self.stack) == 0

  def copy(self):
    r = RnngDecoderState(stack_lstm_state=self.stack_lstm_state,
                         term_lstm_state=self.term_lstm_state,
                         action_lstm_state=self.action_lstm_state)
    # We need to create a copy of the "children" bits of the
    # RnngStackElements instead of just a reference to the same list.
    for nt_id, children, prev_state in self.stack:
      r.stack.append(RnngStackElement(nt_id, children[:], prev_state))
    r.terminals = self.terminals[:]
    r.actions = self.actions[:]
    return r

class RnngDecoderStateBatch(batchers.ListBatch):
  def __init__(self, batch_elements):
    super().__init__(batch_elements)

  def as_vector(self):
    return dy.concatenate_to_batch([elem.as_vector() for elem in self])

  def is_forbidden(self, word):
    if not batchers.is_batched(word):
      word = batchers.ListBatch([word] * self.batch_size())

    assert self.batch_size() == word.batch_size()
    r = []
    for i in range(self.batch_size()):
      r.append(self[i].is_forbidden(word[i]))
    r = batchers.mark_as_batch(r)
    assert self.batch_size() == r.batch_size()
    return r

  def is_complete(self):
    return all([elem.is_complete() for elem in self])

class RnngDecoderBase(Decoder, Serializable):
  yaml_tag = '!RnngDecoderBase'

  @serializable_init
  def __init__(self,
               input_dim = Ref("exp_global.default_layer_dim"),
               hidden_dim = Ref("exp_global.default_layer_dim"),
               embedder: embedders.Embedder = bare(embedders.RnngEmbedder),
               action_scorer=bare(scorers.Softmax, vocab_size=RnngVocab.NUM_ACTIONS),
               term_scorer = bare(scorers.Softmax),
               nt_scorer = bare(scorers.Softmax),
               bridge: bridges.Bridge = bare(bridges.CopyBridge),
               vocab=None,
               term_lstm=None,
               action_lstm=None,
               stack_lstm=bare(recurrent.UniLSTMSeqTransducer, decoder_input_feeding=False),
               state_transform=bare(transforms.AuxNonLinear),
               word_emb_transform=bare(transforms.Linear, bias=False),
               use_term_lstm=False,
               use_action_lstm=False):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.embedder = embedder
    self.action_scorer = action_scorer
    self.term_scorer = term_scorer
    self.nt_scorer = nt_scorer
    self.bridge = bridge

    self.vocab = vocab

    self.use_term_lstm = use_term_lstm
    self.use_action_lstm = use_action_lstm

    # LSTMs
    if use_term_lstm:
      self.term_lstm = self.add_serializable_component("term_lstm", term_lstm,
                                                       lambda: recurrent.UniLSTMSeqTransducer(input_dim=self.input_dim, hidden_dim=self.hidden_dim, decoder_input_feeding=False))
    if use_action_lstm:
      self.action_lstm = self.add_serializable_component("action_lstm", action_lstm,
                                                         lambda: recurrent.UniLSTMSeqTransducer(input_dim=self.input_dim, hidden_dim=self.hidden_dim, decoder_input_feeding=False))
    self.stack_lstm = stack_lstm

    # Transform the word embeddings from embedder.emb_dim into hidden_dim
    self.word_emb_transform = word_emb_transform

    # The parser state is computed as
    # tanh(W * [s; w] + b)
    # where s is the output of the stack LSTM
    # and w is the source context
    self.state_transform = state_transform

  def calc_state(self, dec_state : RnngDecoderState):
    s = dec_state.as_vector()
    c = dec_state.context
    if s.dim()[1] != c.dim()[1]:
      s2 = [s for _ in range(c.dim()[1])]
      s = dy.concatenate_to_batch(s2)
    if s.dim()[0] != c.dim()[1]:
      c = dy.reshape(c, s.dim()[0])
    assert s.dim() == c.dim(), 'state dim is %s but should be %s' % (str(s.dim()), str(c.dim()))
    state_in = dy.concatenate([s, c])
    state = self.state_transform.transform(state_in)
    return state

  def calc_loss(self, dec_state : RnngDecoderState, ref_action : RnngAction):
    assert dec_state.context != None
    assert type(ref_action) == batchers.ListBatch
    batched = batchers.is_batched(ref_action)

    if not batched:
      assert len(ref_action) == 1
      ref_action = ref_action[0]

    state = self.calc_state(dec_state)

    ref_action_type = ref_action[0] if not batched else batchers.ListBatch([r[0] for r in ref_action])
    ref_action_subtype = ref_action[1] if not batched else batchers.ListBatch([r[1] for r in ref_action])

    action_loss = self.action_scorer.calc_loss(state, ref_action_type)
    loss = action_loss
    if not batched:
      subloss = self.calc_subloss(state, ref_action_type, ref_action_subtype)
    else:
      subloss = self.calc_subloss_batch(state, ref_action_type, ref_action_subtype)
    loss = action_loss + subloss

    if batched:
      mask = dy.inputTensor([1 if r != RnngVocab.NONE else 0 for r in ref_action_type], batched=True)
      loss = dy.cmult(loss, mask)

    return loss

  def calc_subloss(self, state, ref_action_type, ref_action_subtype):
    assert state.dim()[1] == 1
    assert not batchers.is_batched(ref_action_type)
    assert not batchers.is_batched(ref_action_subtype)
    if ref_action_type == RnngVocab.SHIFT:
      term_loss = self.term_scorer.calc_loss(state, ref_action_subtype)
      return term_loss
    elif ref_action_type == RnngVocab.NT:
      nt_loss = self.nt_scorer.calc_loss(state, ref_action_subtype)
      return nt_loss
    return dy.zeros((1,))

  def calc_subloss_batch(self, state: dy.Expression, ref_action_type: batchers.ListBatch, ref_action_subtype: batchers.ListBatch):
    N = len(ref_action_type)
    used_action_types = set(ref_action_type)
    subloss = dy.zeros(1, batch_size=N)
    for action_type in used_action_types:
      if action_type == RnngVocab.SHIFT or action_type == RnngVocab.NT:
        mask = dy.inputTensor([1 if r == action_type else 0 for r in ref_action_type], batched=True)
        subtypes = batchers.ListBatch([ref_action_subtype[i] if ref_action_type[i] == action_type else 0 for i in range(N)])
        scorer = self.term_scorer if action_type == RnngVocab.SHIFT else self.nt_scorer
        scores = scorer.calc_loss(state, subtypes)
        for s, m in zip(scores.npvalue().flatten(), mask.npvalue().flatten()):
          assert np.isfinite(s), '%s\n%s' % (str(scores.npvalue().flatten()), str(mask.npvalue().flatten()))
        subloss += dy.cmult(scores, mask)
    return subloss

  def calc_log_probs(self, dec_state : RnngDecoderState):
    raise NotImplementedError()

  def calc_score(self, calc_scores_logsoftmax : dy.Expression):
    raise NotImplementedError()

  def calc_prob(self, calc_scores_logsoftmax : dy.Expression):
    raise NotImplementedError()

  def calc_log_prob(self, calc_scores_logsoftmax : dy.Expression):
    raise NotImplementedError()

  def lstm_push(self, lstm: recurrent.UniLSTMSeqTransducer, state: recurrent.UniLSTMState, word_emb: dy.Expression):
    c, h = lstm.add_input_to_prev(state, word_emb)
    r = recurrent.UniLSTMState(lstm, state, c=c, h=h)
    return r

  def stack_lstm_push(self,
                      stack_lstm_state: recurrent.UniLSTMState,
                      word_emb: dy.Expression):
    return self.lstm_push(self.stack_lstm, stack_lstm_state, word_emb)

  def term_lstm_push(self,
                     term_lstm_state: recurrent.UniLSTMState,
                     word_emb: dy.Expression):
    assert self.use_term_lstm
    return self.lstm_push(self.term_lstm, term_lstm_state, word_emb)

  def action_lstm_push(self,
                       action_lstm_state: recurrent.UniLSTMState,
                       word_emb: dy.Expression):
    assert self.use_action_lstm
    return self.lstm_push(self.action_lstm, action_lstm_state, word_emb)

  def perform_shift(self, dec_state : RnngDecoderState, word_id : numbers.Integral):
    assert not dec_state.is_forbidden(RnngAction(RnngVocab.SHIFT, word_id))
    assert len(dec_state.stack) > 0
    word_emb = self.embedder.embed_terminal(word_id)
    word_emb = self.word_emb_transform.transform(word_emb)
    #if random.random() < 0.1:
    #  word_emb *= 0

    new_state = dec_state.copy()
    new_state.stack[-1].children.append(word_emb)
    new_state.stack_lstm_state = self.stack_lstm_push(dec_state.stack_lstm_state, word_emb)

    new_state.terminals.append(word_id)
    if self.use_term_lstm:
      new_state.term_lstm_state = self.term_lstm_push(dec_state.term_lstm_state, word_emb)

    if self.use_action_lstm:
      new_state.actions.append(RnngAction(RnngVocab.SHIFT, word_id))
      new_state.action_lstm_state = self.action_lstm_push(dec_state.action_lstm_state, word_emb)
    return new_state

  def perform_nt(self, dec_state, nt_id):
    assert not dec_state.is_forbidden(RnngAction(RnngVocab.NT, nt_id))
    nt_emb = self.embedder.embed_nt(nt_id)
    nt_emb = self.word_emb_transform.transform(nt_emb)
    #if random.random() < 0.1:
    #  nt_emb *= 0

    new_state = dec_state.copy()
    new_state.stack.append(RnngStackElement(nt_id, [], dec_state.stack_lstm_state))
    new_state.stack_lstm_state = self.stack_lstm_push(dec_state.stack_lstm_state, nt_emb)

    if self.use_action_lstm:
      new_state.actions.append(RnngAction(RnngVocab.NT, nt_id))
      new_state.action_lstm_state = self.action_lstm_push(dec_state.action_lstm_state, nt_emb)
    return new_state

  def compose(self, nt_emb, children):
    raise NotImplementedError()

  def perform_reduce(self, dec_state : RnngDecoderState):
    action = RnngAction(RnngVocab.REDUCE, None)
    assert not dec_state.is_forbidden(action)
    assert len(dec_state.stack) > 0

    new_state = dec_state.copy()
    nt_id, children, prev_state = new_state.stack.pop()
    nt_emb = self.embedder.embed_nt(nt_id)
    nt_emb = self.word_emb_transform.transform(nt_emb)

    composed = self.compose(nt_emb, children)

    # For the last reduce of the sentence there will be no more stack
    if len(new_state.stack) > 0:
      new_state.stack[-1].children.append(composed)
    new_state.stack_lstm_state = self.stack_lstm_push(prev_state, composed)

    if self.use_action_lstm:
      reduce_emb = self.embedder.embed(action)
      reduce_emb = self.word_emb_transform.transform(reduce_emb)
      new_state.actions.append(action)
      new_state.action_lstm_state = self.action_lstm_push(dec_state.action_lstm_state, reduce_emb)
    return new_state

  def add_input(self, dec_state : RnngDecoderState, word : RnngAction):
    if dec_state.as_vector().dim()[1] > 1:
      assert dec_state.as_vector().dim()[1] == dec_state.context.dim()[1]
      assert word.batch_size() == dec_state.as_vector().dim()[1]
      assert word.batch_size() == dec_state.context.dim()[1]

    if batchers.is_batched(word):
      new_states = []
      for i in range(word.batch_size()):
        ds = dec_state[i] if dec_state.as_vector().dim()[1] > 1 else dec_state
        new_state = self.add_input(ds, word[i])
        new_states.append(new_state)
      new_states = RnngDecoderStateBatch(new_states)
      return new_states

    if type(dec_state) == RnngDecoderStateBatch:
      assert dec_state.batch_size() == 1
      dec_state = dec_state[0]
      assert type(dec_state) != RnngDecoderStateBatch

    if dec_state.is_forbidden(word):
      if type(dec_state) == RnngDecoderStateBatch:
        for s in dec_state:
          print('stack: %d, terms: %d' % (len(s.stack), len(s.terminals)))
      else:
        s = dec_state
        print('stack: %d, terms: %d' % (len(s.stack), len(s.terminals)))
      print('%s is forbidden in dec_state above.' % (self.vocab[word]))
      print(dec_state.is_forbidden(word))

    assert not dec_state.is_forbidden(word), '%s (type %s) is forbidden in dec_state %s (type %s)' % (str(word), str(type(word)), str(dec_state), str(type(dec_state)))
    assert not dec_state.is_complete() or word.action == RnngVocab.NONE, 'Attempt to add to a complete hypothesis!'
    if word.action == RnngVocab.SHIFT:
      return self.perform_shift(dec_state, word.subaction)
    elif word.action == RnngVocab.NT:
      return self.perform_nt(dec_state, word.subaction)
    elif word.action == RnngVocab.REDUCE:
      return self.perform_reduce(dec_state)
    elif word.action == RnngVocab.NONE:
      return dec_state.copy()
    else:
      raise Exception()

  def initial_state(self, enc_final_states, ss_expr):
    stack_lstm_state = self.stack_lstm.initial_state()
    term_lstm_state = self.term_lstm.initial_state() if self.use_term_lstm else None
    action_lstm_state = self.action_lstm.initial_state() if self.use_action_lstm else None
    return RnngDecoderState(stack_lstm_state, term_lstm_state, action_lstm_state)

  def best_k(self, dec_state: RnngDecoderState, k: numbers.Integral, normalize_scores: bool = False):
    state = self.calc_state(dec_state)

    action_log_probs = self.action_scorer.calc_log_probs(state).npvalue()
    best_terms = self.term_scorer.best_k(state, k, normalize_scores=True)
    best_nts = self.nt_scorer.best_k(state, k, normalize_scores=True)

    shift_score = action_log_probs[RnngVocab.SHIFT]
    nt_score = action_log_probs[RnngVocab.NT]
    reduce_score = action_log_probs[RnngVocab.REDUCE]

    best_actions = []
    # Add best SHIFT actions
    for term, score in zip(*best_terms):
      assert term < len(self.vocab.term_vocab)
      action = RnngAction(RnngVocab.SHIFT, term)
      total_score = shift_score + score
      if not dec_state.is_forbidden(action):
        heapq.heappush(best_actions, (-total_score, action))

    # Add best NT actions
    for nt, score in zip(*best_nts):
      assert nt < len(self.vocab.nt_vocab), 'Attempt to get element %d from NT vocab of size %d' % (nt, len(self.vocab.nt_vocab))
      action = RnngAction(RnngVocab.NT, nt)
      total_score = nt_score + score
      if not dec_state.is_forbidden(action):
        heapq.heappush(best_actions, (-total_score, action))

    # Add REDUCE action
    if False:
      action = RnngAction(RnngVocab.REDUCE, None)
      total_score = reduce_score
      if not dec_state.is_forbidden(action):
        heapq.heappush(best_actions, (-total_score, action))

    r_actions = []
    r_scores = []
    while len(r_actions) < k and len(best_actions) > 0:
      score, action = heapq.heappop(best_actions)
      if not dec_state.is_forbidden(action):
        r_actions.append(action)
        r_scores.append(-score)
    return r_actions, r_scores

  def sample(self, dec_state: RnngDecoderState, n: numbers.Integral, temperature: float = 1.0):
    assert n == 1
    state = self.calc_state(dec_state)
    actions = self.action_scorer.sample(state, n, temperature)
    action = RnngAction(actions[0][0], 0)

    # Some hack thing to not produce invalid actions
    tries = 0
    fb = dec_state.is_forbidden(action)
    while fb == True or (batchers.is_batched(fb) and any(fb)) or action.action == 3:
      actions = self.action_scorer.sample(state, n, temperature)
      action = RnngAction(actions[0][0], 0)
      fb = dec_state.is_forbidden(action)
      tries += 1
      assert tries < 100

    if actions[0][0] == RnngVocab.SHIFT:
      subaction = self.term_scorer.sample(state, n)
      return [(RnngAction(RnngVocab.SHIFT, subaction[0][0]), actions[0][1] + subaction[0][1])]
    elif actions[0][0] == RnngVocab.NT:
      subaction = self.nt_scorer.sample(state, n)
      return [(RnngAction(RnngVocab.NT, subaction[0][0]), actions[0][1] + subaction[0][1])]
    else:
      assert False

  def shared_params(self):
    return [{".embedder.nt_emb.emb_dim", ".embedder.term_emb.emb_dim"},
            {".embedder.nt_emb.emb_dim", ".word_emb_transform.input_dim"},
            {".hidden_dim", ".word_emb_transform.output_dim"},
            {".hidden_dim", ".action_scorer.input_dim"},
            {".hidden_dim", ".stack_lstm.input_dim"},
            {".hidden_dim", ".stack_lstm.hidden_dim"},
            {".hidden_dim", ".state_transform.input_dim"},
            {".input_dim", ".state_transform.aux_input_dim"},
            {".hidden_dim", ".state_transform.output_dim"},
            {".hidden_dim", ".nt_scorer.input_dim"},
            {".hidden_dim", ".term_scorer.input_dim"}]

class RnngDecoder(RnngDecoderBase, Serializable):
  yaml_tag = "!RnngDecoder"

  @serializable_init
  def __init__(self, input_dim = Ref("exp_global.default_layer_dim"),
               hidden_dim = Ref("exp_global.default_layer_dim"),
               embedder: embedders.Embedder = bare(embedders.RnngEmbedder),
               action_scorer=bare(scorers.Softmax, vocab_size=RnngVocab.NUM_ACTIONS),
               term_scorer = bare(scorers.Softmax),
               nt_scorer = bare(scorers.Softmax),
               bridge: bridges.Bridge = bare(bridges.CopyBridge),
               vocab=None,
               term_lstm=None,
               action_lstm=None,
               stack_lstm=bare(recurrent.UniLSTMSeqTransducer, decoder_input_feeding=False),
               state_transform=bare(transforms.AuxNonLinear),
               word_emb_transform=bare(transforms.Linear, bias=False),
               use_term_lstm=False,
               use_action_lstm=False,
               comp_lstm_fwd=bare(recurrent.UniLSTMSeqTransducer, decoder_input_feeding=False),
               comp_lstm_rev=bare(recurrent.UniLSTMSeqTransducer, decoder_input_feeding=False),
               compose_transform=bare(transforms.NonLinear)):
    super().__init__(input_dim, hidden_dim, embedder, action_scorer, term_scorer, nt_scorer, bridge, vocab, term_lstm, action_lstm, stack_lstm, state_transform, word_emb_transform, use_term_lstm, use_action_lstm)

    # Composed representation of a treelet is composed as follows:
    # f = LSTM(label + children)
    # r = LSTM(label + children[::-1])
    # final = tanh(w * [f; r] + b) = tanh(linear([f; r]))
    self.comp_lstm_fwd = comp_lstm_fwd
    self.comp_lstm_rev = comp_lstm_rev
    self.compose_transform = compose_transform

  def compose(self, nt_emb, children):
    fwd_children = expression_seqs.ExpressionSequence([nt_emb] + children)
    rev_children = expression_seqs.ExpressionSequence([nt_emb] + children[::-1])
    fwd_lstm_out = self.comp_lstm_fwd.transduce([fwd_children])[-1]
    rev_lstm_out = self.comp_lstm_rev.transduce([rev_children])[-1]
    bidir_out = fwd_lstm_out + rev_lstm_out
    composed = self.compose_transform.transform(bidir_out)
    return composed

  def shared_params(self):
    return super().shared_params() + \
        [{".hidden_dim", ".comp_lstm_fwd.input_dim"},
         {".hidden_dim", ".comp_lstm_rev.input_dim"},
         {".hidden_dim", ".comp_lstm_fwd.hidden_dim"},
         {".hidden_dim", ".comp_lstm_rev.hidden_dim"},
         {".hidden_dim", ".compose_transform.input_dim"},
         {".hidden_dim", ".compose_transform.output_dim"}]

class BinaryRnngDecoder(RnngDecoderBase, Serializable):
  yaml_tag = "!BinaryRnngDecoder"

  @serializable_init
  def __init__(self, input_dim = Ref("exp_global.default_layer_dim"),
               hidden_dim = Ref("exp_global.default_layer_dim"),
               embedder: embedders.Embedder = bare(embedders.RnngEmbedder),
               action_scorer=bare(scorers.Softmax, vocab_size=RnngVocab.NUM_ACTIONS),
               term_scorer = bare(scorers.Softmax),
               nt_scorer = bare(scorers.Softmax),
               bridge: bridges.Bridge = bare(bridges.CopyBridge),
               vocab=None,
               term_lstm=None,
               action_lstm=None,
               stack_lstm=bare(recurrent.UniLSTMSeqTransducer, decoder_input_feeding=False),
               state_transform=bare(transforms.AuxNonLinear),
               word_emb_transform=bare(transforms.Linear, bias=False),
               use_term_lstm=False,
               use_action_lstm=False,
               compose_transform=None):
    super().__init__(input_dim, hidden_dim, embedder, action_scorer, term_scorer, nt_scorer, bridge, vocab, term_lstm, action_lstm, stack_lstm, state_transform, word_emb_transform, use_term_lstm, use_action_lstm)

    # Composed representation of a treelet is composed as follows:
    # final = tanh(u * label + w * child1 + v * child2 + b)
    self.compose_transform = self.add_serializable_component(
        "compose_transform", compose_transform,
        lambda: transforms.NonLinear(input_dim=3*self.hidden_dim, output_dim=self.hidden_dim))

  def compose(self, nt_emb, children):
    assert len(children) > 0

    # If we have only one child, fill in zeros for the other child
    if len(children) == 1:
      children.append(dy.zeros(children[0].dim()[0]))
    assert len(children) == 2, 'Expected exactly two children but got %d' % len(children)

    # Concatenate the label and the two children and transform
    c = dy.concatenate([nt_emb] + children)
    return self.compose_transform.transform(c)

  def perform_shift(self, dec_state : RnngDecoderState, word_id : numbers.Integral):
    new_state = super().perform_shift(dec_state, word_id)
    while len(new_state.stack) > 0 and len(new_state.stack[-1].children) >= 2:
      new_state = self.perform_reduce(new_state)
    return new_state
