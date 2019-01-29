from typing import Any
from collections import namedtuple
import numbers

import dynet as dy

from xnmt import batchers, param_collections, expression_seqs
from xnmt.modelparts import bridges, transforms, scorers, embedders
from xnmt.transducers import recurrent
from xnmt.persistence import serializable_init, Serializable, bare, Ref
from xnmt.vocabs import RnngVocab, RnngAction

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

class AutoRegressiveDecoderState(DecoderState):
  """A state holding all the information needed for AutoRegressiveDecoder
  
  Args:
    rnn_state: a DyNet RNN state
    context: a DyNet expression
  """
  def __init__(self, rnn_state=None, context=None):
    self.rnn_state = rnn_state
    self.context = context

  def as_vector(self):
    return self.rnn_state.output()

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
    zeros = dy.zeros(self.input_dim) if self.input_feeding else None
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
    trg_embedding = self.embedder.embed(trg_word)
    inp = trg_embedding
    if self.input_feeding:
      inp = dy.concatenate([inp, mlp_dec_state.context])
    rnn_state = mlp_dec_state.rnn_state
    if self.truncate_dec_batches: rnn_state, inp = batchers.truncate_batches(rnn_state, inp)
    return AutoRegressiveDecoderState(rnn_state=rnn_state.add_input(inp),
                                      context=mlp_dec_state.context)

  def _calc_transform(self, mlp_dec_state: AutoRegressiveDecoderState) -> dy.Expression:
    h = dy.concatenate([mlp_dec_state.rnn_state.output(), mlp_dec_state.context])
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

  def __init__(self, stack_emb=None):
    self.stack = [] # Stack of RnngStackElements
    self.terminals = [] # Generated terminals
    self.stack_emb = stack_emb
    self.context = None

  def __str__(self):
    return '\n'.join(['stack: ' + str(self.stack), 'terms:' + str(self.terminals)])

  def is_forbidden(self, a):
    if a.action not in [RnngVocab.SHIFT, RnngVocab.NT, RnngVocab.REDUCE]:
      return True

    if len(self.stack) >= RnngDecoderState.MaxOpenNTs:
        return True

    if len(self.stack) == 0:
      if a.action != RnngVocab.NT:
        return True

    if a.action == RnngVocab.REDUCE:
      if len(self.stack) == 0:
        return True
      if len(self.stack[-1].children) == 0:
        return True

    return False

  def as_vector(self):
    return self.stack_emb.output()

  def copy(self):
    r = RnngDecoderState(stack_emb=self.stack_emb)
    # We need to create a copy of the "children" bits of the
    # RnngStackElements instead of just a reference to the same list.
    for nt_id, children, prev_state in self.stack:
      r.stack.append(RnngStackElement(nt_id, children[:], prev_state))
    r.terminals = self.terminals[:]
    return r

class RnngDecoder(Decoder, Serializable):
  yaml_tag = "!RnngDecoder"

  @serializable_init
  def __init__(self, input_dim = Ref("exp_global.default_layer_dim"),
               hidden_dim = Ref("exp_global.default_layer_dim"),
               dropout = 0.0,
               embedder: embedders.Embedder = bare(embedders.RnngEmbedder), 
               action_scorer=bare(scorers.Softmax, vocab_size=RnngVocab.NUM_ACTIONS),
               term_scorer = bare(scorers.Softmax), nt_scorer = bare(scorers.Softmax),
               bridge: bridges.Bridge = bare(bridges.CopyBridge),
               vocab=None,
               stack_lstm=bare(recurrent.UniLSTMSeqTransducer, decoder_input_feeding=False),
               comp_lstm_fwd=bare(recurrent.UniLSTMSeqTransducer, decoder_input_feeding=False),
               comp_lstm_rev=bare(recurrent.UniLSTMSeqTransducer, decoder_input_feeding=False),
               compose_transform=bare(transforms.NonLinear),
               state_transform=bare(transforms.AuxNonLinear),
               word_emb_transform=bare(transforms.Linear, bias=False)):

    #model = param_collections.ParamManager.my_params(self)
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    self.embedder = embedder
    self.action_scorer = action_scorer
    self.term_scorer = term_scorer
    self.nt_scorer = nt_scorer
    self.bridge = bridge

    self.vocab = vocab

    # LSTMs
    self.stack_lstm = stack_lstm 
    self.comp_lstm_fwd = comp_lstm_fwd
    self.comp_lstm_rev = comp_lstm_rev


    # Transform the word embeddings from input_dim into hidden_dim
    self.word_emb_transform = word_emb_transform

    # Composed representation of a treelet is composed as follows:
    # f = LSTM(label + children)
    # r = LSTM(label + children[::-1])
    # final = tanh(w * [f; r] + b) = tanh(linear([f; r]))
    self.compose_transform = compose_transform

    # The parser state is computed as
    # tanh(W * [s; w] + b)
    # where s is the output of the stack LSTM
    # and w is the source context
    self.state_transform = state_transform

  def calc_loss(self, dec_state, ref_action):
    assert dec_state.context != None
    assert type(ref_action) == batchers.ListBatch
    assert len(ref_action) == 1
    ref_action = ref_action[0]

    state_in = dy.concatenate([dec_state.as_vector(), dec_state.context])
    state = self.state_transform.transform(state_in)

    #action_log_probs = self.action_scorer.calc_log_probs(state)
    #action_log_prob = dy.pick(action_log_probs, ref_action[0])
    #log_prob = action_log_prob
    action_loss = self.action_scorer.calc_loss(state, ref_action[0])
    loss = action_loss
    if ref_action[0] == RnngVocab.SHIFT:
      #term_log_probs = self.term_scorer.calc_log_probs(state)
      #term_log_prob = dy.pick(term_log_probs, ref_action[1])
      #log_prob += term_log_prob
      term_loss = self.term_scorer.calc_loss(state, ref_action[1])
      loss += term_loss
    elif ref_action[0] == RnngVocab.NT:
      #nt_log_probs = self.nt_scorer.calc_log_probs(state)
      #nt_log_prob = dy.pick(nt_log_probs, ref_action[1])
      #log_prob += nt_log_prob
      nt_loss = self.nt_scorer.calc_loss(state, ref_action[1])
      loss += nt_loss
    return loss
    return -log_prob

  def calc_log_probs(self, dec_state):
    raise NotImplementedError()

  def calc_score(self, calc_scores_logsoftmax):
    raise NotImplementedError()

  def calc_prob(self, calc_scores_logsoftmax):
    raise NotImplementedError()

  def calc_log_prob(self, calc_scores_logsoftmax):
    raise NotImplementedError()

  def stack_lstm_push(self, state_emb, word_emb):
    c, h = self.stack_lstm.add_input_to_prev(state_emb, word_emb)
    r = recurrent.UniLSTMState(self.stack_lstm, state_emb, c=c, h=h)
    return r 

  def perform_shift(self, dec_state, word_id):
    assert not dec_state.is_forbidden(RnngAction(RnngVocab.SHIFT, word_id))
    assert len(dec_state.stack) > 0
    word_emb = self.embedder.embed_terminal(word_id)
    word_emb = self.word_emb_transform.transform(word_emb)

    new_state = dec_state.copy()
    new_state.stack[-1].children.append(word_emb)
    new_state.stack_emb = self.stack_lstm_push(dec_state.stack_emb, word_emb)
    new_state.terminals.append(word_id)
    return new_state

  def perform_nt(self, dec_state, nt_id):
    assert not dec_state.is_forbidden(RnngAction(RnngVocab.NT, nt_id)) 
    nt_emb = self.embedder.embed_nt(nt_id)

    new_state = dec_state.copy()
    new_state.stack.append(RnngStackElement(nt_id, [], dec_state.stack_emb))
    new_state.stack_emb = self.stack_lstm_push(dec_state.stack_emb, nt_emb)
    return new_state

  def perform_reduce(self, dec_state):
    assert not dec_state.is_forbidden(RnngAction(RnngVocab.REDUCE, None))
    assert len(dec_state.stack) > 0

    new_state = dec_state.copy()
    nt_id, children, prev_state = new_state.stack.pop()
    nt_emb = self.embedder.embed_nt(nt_id)

    fwd_children = expression_seqs.ExpressionSequence([nt_emb] + children)
    rev_children = expression_seqs.ExpressionSequence([nt_emb] + children[::-1])
    fwd_lstm_out = self.comp_lstm_fwd.transduce([fwd_children])[-1]
    rev_lstm_out = self.comp_lstm_rev.transduce([rev_children])[-1]
    bidir_out = fwd_lstm_out + rev_lstm_out
    composed = self.compose_transform.transform(bidir_out)

    # For the last reduce of the sentence there will be no more stack
    if len(new_state.stack) > 0:
      new_state.stack[-1].children.append(composed)
    new_state.stack_emb = self.stack_lstm_push(prev_state, composed)
      
    return new_state

  def add_input(self, dec_state, word):
    assert len(word) == 1
    word = word[0]
    assert not dec_state.is_forbidden(word)

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
    return RnngDecoderState(self.stack_lstm.initial_state())

  def shared_params(self):
    return [{".input_dim", ".embedder.term_emb.emb_dim"},
            {".input_dim", ".embedder.nt_emb.emb_dim"},
            {".input_dim", ".word_emb_transform.input_dim"},
            {".hidden_dim", ".word_emb_transform.output_dim"},
            {".hidden_dim", ".action_scorer.input_dim"},
            {".hidden_dim", ".stack_lstm.input_dim"},
            {".hidden_dim", ".comp_lstm_fwd.input_dim"},
            {".hidden_dim", ".comp_lstm_rev.input_dim"},
            {".hidden_dim", ".stack_lstm.hidden_dim"},
            {".hidden_dim", ".comp_lstm_fwd.hidden_dim"},
            {".hidden_dim", ".comp_lstm_rev.hidden_dim"},
            {".hidden_dim", ".state_transform.input_dim"},
            {".hidden_dim", ".state_transform.aux_input_dim"},
            {".hidden_dim", ".state_transform.output_dim"},
            {".hidden_dim", ".compose_transform.input_dim"},
            {".hidden_dim", ".compose_transform.output_dim"}]

# TODO: This should be factored to simply use Softmax
# class AutoRegressiveLexiconDecoder(AutoRegressiveDecoder, Serializable):
#   yaml_tag = '!AutoRegressiveLexiconDecoder'
# 
#   @register_xnmt_handler
#   @serializable_init
#   def __init__(self,
#                input_dim=Ref("exp_global.default_dim"),
#                trg_embed_dim=Ref("exp_global.default_dim"),
#                input_feeding=True,
#                rnn=bare(UniLSTMSeqTransducer),
#                mlp=bare(AttentionalOutputMLP),
#                bridge=bare(CopyBridge),
#                label_smoothing=0.0,
#                lexicon_file=None,
#                src_vocab=Ref(Path("model.src_reader.vocab")),
#                trg_vocab=Ref(Path("model.trg_reader.vocab")),
#                attender=Ref(Path("model.attender")),
#                lexicon_type='bias',
#                lexicon_alpha=0.001,
#                linear_projector=None,
#                truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False),
#                param_init_lin=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
#                bias_init_lin=Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
#                ) -> None:
#     super().__init__(input_dim, trg_embed_dim, input_feeding, rnn,
#                      mlp, bridge, truncate_dec_batches, label_smoothing)
#     assert lexicon_file is not None
#     self.lexicon_file = lexicon_file
#     self.src_vocab = src_vocab
#     self.trg_vocab = trg_vocab
#     self.attender = attender
#     self.lexicon_type = lexicon_type
#     self.lexicon_alpha = lexicon_alpha
# 
#     self.linear_projector = self.add_serializable_component("linear_projector", linear_projector,
#                                                              lambda: xnmt.linear.Linear(input_dim=input_dim,
#                                                                                         output_dim=mlp.output_dim))
# 
#     if self.lexicon_type == "linear":
#       self.lexicon_method = self.linear
#     elif self.lexicon_type == "bias":
#       self.lexicon_method = self.bias
#     else:
#       raise ValueError("Unrecognized lexicon method:", lexicon_type, "can only choose between [bias, linear]")
# 
#   def load_lexicon(self):
#     logger.info("Loading lexicon from file: " + self.lexicon_file)
#     assert self.src_vocab.frozen
#     assert self.trg_vocab.frozen
#     lexicon = [{} for _ in range(len(self.src_vocab))]
#     with open(self.lexicon_file, encoding='utf-8') as fp:
#       for line in fp:
#         try:
#           trg, src, prob = line.rstrip().split()
#         except:
#           logger.warning("Failed to parse 'trg src prob' from:" + line.strip())
#           continue
#         trg_id = self.trg_vocab.convert(trg)
#         src_id = self.src_vocab.convert(src)
#         lexicon[src_id][trg_id] = float(prob)
#     # Setting the rest of the weight to the unknown word
#     for i in range(len(lexicon)):
#       sum_prob = sum(lexicon[i].values())
#       if sum_prob < 1.0:
#         lexicon[i][self.trg_vocab.convert(self.trg_vocab.unk_token)] = 1.0 - sum_prob
#     # Overriding special tokens
#     src_unk_id = self.src_vocab.convert(self.src_vocab.unk_token)
#     trg_unk_id = self.trg_vocab.convert(self.trg_vocab.unk_token)
#     lexicon[self.src_vocab.SS] = {self.trg_vocab.SS: 1.0}
#     lexicon[self.src_vocab.ES] = {self.trg_vocab.ES: 1.0}
#     # TODO(philip30): Note sure if this is intended
#     lexicon[src_unk_id] = {trg_unk_id: 1.0}
#     return lexicon
# 
#   @handle_xnmt_event
#   def on_new_epoch(self, training_task, *args, **kwargs):
#     if hasattr(self, "lexicon_prob"):
#       del self.lexicon_prob
#     if not hasattr(self, "lexicon"):
#       self.lexicon = self.load_lexicon()
# 
#   @handle_xnmt_event
#   def on_start_sent(self, src):
#     batch_size = len(src)
#     col_size = len(src[0])
# 
#     idxs = [(x, j, i) for i in range(batch_size) for j in range(col_size) for x in self.lexicon[src[i][j]].keys()]
#     idxs = tuple(map(list, list(zip(*idxs))))
# 
#     values = [x for i in range(batch_size) for j in range(col_size) for x in self.lexicon[src[i][j]].values()]
#     self.lexicon_prob = dy.nobackprop(dy.sparse_inputTensor(idxs, values, (len(self.trg_vocab), col_size, batch_size), batched=True))
#     
#   def calc_scores_logsoftmax(self, mlp_dec_state):
#     score = super().calc_scores(mlp_dec_state)
#     lex_prob = self.lexicon_prob * self.attender.get_last_attention()
#     # Note that the sum dim is only summing a tensor of 1 size in dim 1.
#     # This is to make sure that the shape of the returned tensor matches the vanilla decoder
#     return dy.sum_dim(self.lexicon_method(mlp_dec_state, score, lex_prob), [1])
# 
#   def linear(self, mlp_dec_state, score, lex_prob):
#     coef = dy.logistic(self.linear_projector(mlp_dec_state.rnn_state.output()))
#     return dy.log(dy.cmult(dy.softmax(score), coef) + dy.cmult((1-coef), lex_prob))
# 
#   def bias(self, mlp_dec_state, score, lex_prob):
#     return dy.log_softmax(score + dy.log(lex_prob + self.lexicon_alpha))
# 
#   def calc_loss(self, mlp_dec_state, ref_action):
#     logsoft = self.calc_scores_logsoftmax(mlp_dec_state)
#     if not xnmt.batcher.is_batched(ref_action):
#       return -dy.pick(logsoft, ref_action)
#     else:
#       return -dy.pick_batch(logsoft, ref_action)
