import math
import numbers

import numpy as np
import dynet as dy

from xnmt import logger
from xnmt import batchers, expression_seqs, events, param_collections, param_initializers
from xnmt.modelparts import decoders
from xnmt.persistence import serializable_init, Serializable, Ref, bare
from xnmt.transducers import recurrent
from xnmt.sent import SyntaxTree

class AttenderState(object):
  pass

class Attender(object):
  """
  A template class for functions implementing attention.
  """

  def init_sent(self, encoding: expression_seqs.ExpressionSequence, sent) -> AttenderState:
    """Args:
         encoding: the encoder states, aka keys and values. Usually but not necessarily an :class:`expression_seqs.ExpressionSequence`
         sent: the raw source sentence
       Returns:
         An initial state of the attender, representing not yet having attended to anything.
    """
    raise NotImplementedError('init_sent must be implemented for Attender subclasses')

  def calc_attention(self, dec_state: dy.Expression, att_state: AttenderState = None) -> dy.Expression:
    """ Compute attention weights.

    Args:
      dec_state: the current decoder state, aka query, for which to compute the weights.
      att_state: the current attender state
    Returns:
      DyNet expression containing normalized attention scores
    """
    raise NotImplementedError('calc_attention must be implemented for Attender subclasses')

  def calc_context(self, dec_state: dy.Expression, att_state: AttenderState = None, attention: dy.Expression = None) -> dy.Expression:
    """ Compute weighted sum.

    Args:
      dec_state: the current decoder state, aka query, for which to compute the weighted sum.
      att_state: the current attender state
      attention: the attention vector to use. if not given it is calculated from the state(s).
    """
    attention = attention or self.calc_attention(dec_state, att_state)
    I = self.curr_sent.as_tensor()
    return I * attention

  def update(self, dec_state, att_state: AttenderState, attention: dy.Expression):
    return None

def safe_affine_transform(xs):
  r = dy.affine_transform(xs)
  d = r.dim()
  # TODO(philip30): dynet affine transform bug, should be fixed upstream
  # if the input size is "1" then the last dimension will be dropped.
  if len(d[0]) == 1:
    r = dy.reshape(r, (d[0][0], 1), batch_size=d[1])
  return r

class MlpAttender(Attender, Serializable):
  """
  Implements the attention model of Bahdanau et. al (2014)

  Args:
    input_dim: input dimension
    state_dim: dimension of dec_state inputs
    hidden_dim: hidden MLP dimension
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
  """

  yaml_tag = '!MlpAttender'

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               state_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    self.truncate_dec_batches = truncate_dec_batches
    param_collection = param_collections.ParamManager.my_params(self)
    self.W = param_collection.add_parameters((hidden_dim, input_dim), init=param_init.initializer((hidden_dim, input_dim)))
    self.V = param_collection.add_parameters((hidden_dim, state_dim), init=param_init.initializer((hidden_dim, state_dim)))
    self.b = param_collection.add_parameters((hidden_dim,), init=bias_init.initializer((hidden_dim,)))
    self.U = param_collection.add_parameters((1, hidden_dim), init=param_init.initializer((1, hidden_dim)))
    self.curr_sent = None

  def init_sent(self, encoded: expression_seqs.ExpressionSequence, sent) -> None:
    self.attention_vecs = []
    self.curr_sent = encoded
    I = self.curr_sent.as_tensor()
    self.WI = safe_affine_transform([self.b, self.W, I])
    return None

  def calc_attention(self, dec_state: dy.Expression, att_state: AttenderState = None) -> dy.Expression:
    WI = self.WI
    curr_sent_mask = self.curr_sent.mask
    if self.truncate_dec_batches:
      if curr_sent_mask:
        dec_state, WI, curr_sent_mask = batchers.truncate_batches(dec_state, WI, curr_sent_mask)
      else:
        dec_state, WI = batchers.truncate_batches(dec_state, WI)
    h = dy.tanh(dy.colwise_add(WI, self.V * dec_state))
    scores = dy.transpose(self.U * h)
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

class CoverageGRU(Serializable):
  """A specialization of UniGRUSeqTransducer used by Coverage Attenders.
  A coverage attender updates its attention state by using a GRU with three
  inputs: the current decoder state, the most recent attention weight, and
  a source word vector.
  Logically these three inputs are concatenated and fed into the GRU once
  per source word at each time step. This class saves some time and RAM
  by pre-computing weights * decoder state once per time step and
  weights * source word vectors once per sentence.

  For now this class only supports one-layer GRUs."""

  yaml_tag = '!CoverageGRU'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               src_vec_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dec_state_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               coverage_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))):
    self.src_vec_dim = src_vec_dim
    self.dec_state_dim = dec_state_dim
    self.coverage_dim = coverage_dim
    self.dropout_rate = dropout

    self.dropout_mask = None
    self.create_parameters(param_init, bias_init)

  def create_parameters(self, param_init, bias_init):
    ws_dim = (self.coverage_dim, self.src_vec_dim)
    wd_dim = (self.coverage_dim, self.dec_state_dim)
    wa_dim = (self.coverage_dim, 1)
    u_dim = (self.coverage_dim, self.coverage_dim)
    b_dim = (self.coverage_dim,)

    model = param_collections.ParamManager.my_params(self)
    self.Wsz = model.add_parameters(dim=ws_dim, init=param_init.initializer(ws_dim))
    self.Wdz = model.add_parameters(dim=wd_dim, init=param_init.initializer(wd_dim))
    self.Waz = model.add_parameters(dim=wa_dim, init=param_init.initializer(wa_dim))
    self.Wsr = model.add_parameters(dim=ws_dim, init=param_init.initializer(ws_dim))
    self.Wdr = model.add_parameters(dim=wd_dim, init=param_init.initializer(wd_dim))
    self.War = model.add_parameters(dim=wa_dim, init=param_init.initializer(wa_dim))
    self.Wsu = model.add_parameters(dim=ws_dim, init=param_init.initializer(ws_dim))
    self.Wdu = model.add_parameters(dim=wd_dim, init=param_init.initializer(wd_dim))
    self.Wau = model.add_parameters(dim=wa_dim, init=param_init.initializer(wa_dim))
    self.Uz = model.add_parameters(dim=u_dim, init=param_init.initializer(u_dim))
    self.Ur = model.add_parameters(dim=u_dim, init=param_init.initializer(u_dim))
    self.Uu = model.add_parameters(dim=u_dim, init=param_init.initializer(u_dim))
    self.bz = model.add_parameters(dim=b_dim, init=bias_init.initializer(b_dim))
    self.br = model.add_parameters(dim=b_dim, init=bias_init.initializer(b_dim))
    self.bu = model.add_parameters(dim=b_dim, init=bias_init.initializer(b_dim))

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def initial_state(self) -> dy.Expression:
    return dy.zeros((self.coverage_dim))

  def set_dropout(self, dropout: numbers.Real) -> None:
    self.dropout_rate = dropout

  def set_dropout_masks(self, batch_size: numbers.Integral = 1) -> None:
    # Note: no need for a dropout mask for the input, since it's assumed
    # to be a one-dimensional attention value.
    if self.dropout_rate > 0.0 and self.train:
      retention_rate = 1.0 - self.dropout_rate
      scale = 1.0 / retention_rate
      self.dropout_mask = dy.random_bernoulli((self.converage_dim,), retention_rate, scale, batch_size=batch_size)

  def init_sent(self, src_embs: expression_seqs.ExpressionSequence) -> dy.Expression:
    # Precompute W * src_embs once here
    I = src_embs.as_tensor()
    self.Iz = safe_affine_transform([self.bz, self.Wsz, I])
    self.Ir = safe_affine_transform([self.br, self.Wsr, I])
    self.Iu = safe_affine_transform([self.bu, self.Wsu, I])
    self.dropout_mask = None

    sent_len = I.dim()[0][1]
    batch_size = I.dim()[1]
    return dy.zeros((self.coverage_dim, sent_len), batch_size=batch_size)

  def add_input_to_prev(self, prev_coverage: dy.Expression, dec_state: dy.Expression, attention: dy.Expression):
    iz = safe_affine_transform([self.Iz, self.Waz, dy.transpose(attention), self.Uz, prev_coverage])
    ir = safe_affine_transform([self.Ir, self.War, dy.transpose(attention), self.Ur, prev_coverage])
    iu = safe_affine_transform([self.Iu, self.Wau, dy.transpose(attention), self.Uu, prev_coverage])

    z = dy.logistic(iz)
    r = dy.logistic(ir)
    u = dy.tanh(iu)
    h = dy.cmult(1 - z, prev_coverage) + dy.cmult(z, u)
    return h

class CoverageAttender(Attender, Serializable):
  """
  Implements the attention model of Tu et. al (2016)
  "Modeling Coverage for Neural Machine Translation"

  Args:
    input_dim: input dimension
    state_dim: dimension of dec_state inputs
    hidden_dim: hidden MLP dimension
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
  """

  yaml_tag = '!CoverageAttender'

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               state_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               coverage_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               coverage_rnn = bare(CoverageGRU),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    self.coverage_dim = coverage_dim
    self.coverage_rnn = coverage_rnn
    self.truncate_dec_batches = truncate_dec_batches

    param_collection = param_collections.ParamManager.my_params(self)
    self.W = param_collection.add_parameters((hidden_dim, input_dim), init=param_init.initializer((hidden_dim, input_dim)))
    self.V = param_collection.add_parameters((hidden_dim, state_dim), init=param_init.initializer((hidden_dim, state_dim)))
    self.b = param_collection.add_parameters((hidden_dim,), init=bias_init.initializer((hidden_dim,)))
    self.U = param_collection.add_parameters((hidden_dim, coverage_dim), init=param_init.initializer((hidden_dim, coverage_dim)))
    self.v = param_collection.add_parameters((1, hidden_dim), init=param_init.initializer((1, hidden_dim)))
    self.curr_sent = None

  def init_sent(self, encoded: expression_seqs.ExpressionSequence, sent) -> AttenderState:
    self.attention_vecs = []
    self.curr_sent = encoded
    I = self.curr_sent.as_tensor()
    self.I = I
    self.WI = safe_affine_transform([self.b, self.W, I])
    return self.coverage_rnn.init_sent(encoded)

  def calc_attention(self, dec_state: dy.Expression, att_state: AttenderState = None) -> dy.Expression:
    assert att_state is not None
    WI = self.WI
    curr_sent_mask = self.curr_sent.mask
    if self.truncate_dec_batches:
      if curr_sent_mask:
        dec_state, WI, curr_sent_mask = batchers.truncate_batches(dec_state, WI, curr_sent_mask)
      else:
        dec_state, WI = batchers.truncate_batches(dec_state, WI)

    h_in1 = dy.colwise_add(WI, self.V * dec_state)
    h_in2 = self.U * att_state
    h_in = h_in1 + h_in2
    h = dy.tanh(h_in)
    scores = dy.transpose(self.v * h)
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

  def update(self, dec_state, att_state: AttenderState, attention: dy.Expression):
    return self.coverage_rnn.add_input_to_prev(att_state, dec_state, attention)

  def shared_params(self):
    return [{".coverage_rnn.src_vec_dim", ".input_dim"},
            {".coverage_rnn.dec_state_dim", ".state_dim"},
            {".coverage_rnn.coverage_dim", ".coverage_dim"}]

class SyntaxCoverageAttender(CoverageAttender, Serializable):
  """
  Implements the attention model of Chen et. al (2017)
  "Improved Neural Machine Translation with a Syntax-Aware Encoder and Decoder"

  Args:
    input_dim: input dimension
    state_dim: dimension of dec_state inputs
    hidden_dim: hidden MLP dimension
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
  """

  yaml_tag = '!SyntaxCoverageAttender'

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               state_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               coverage_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               coverage_updater = None,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    super().__init__(input_dim, state_dim, hidden_dim, coverage_dim, coverage_updater, param_init, bias_init, truncate_dec_batches)

  def get_input_dim(self):
    return self.hidden_dim + self.state_dim + 1 + 2 * (self.coverage_dim + 1)

  def build_child_map(self, tree, child_idx):
    child_map = []
    mask = []

    if len(tree.children) > child_idx:
      child = tree.children[child_idx]
      assert child.idx is not None
      child_map.append(child.idx)
      mask.append(1.0)
    else:
      child_map.append(0)
      mask.append(0.0)

    for child in tree.children:
      c, m = self.build_child_map(child, child_idx)
      child_map += c
      mask += m

    return child_map, mask

  def transpose(self, a):
    if len(a) == 0:
      return []

    r = []
    for i in range(len(a[0])):
      r.append([row[i] for row in a])
    return r

  def init_sent(self, encoded: expression_seqs.ExpressionSequence, sent) -> None:
    if type(sent) == batchers.ListBatch:
      assert sent.batch_size() == 1
      sent = sent[0]
    if type(sent) == SyntaxTree:
      sent = batchers.SyntaxTreeBatcher()._make_src_batch([sent])

    self.left_child_map = []
    self.right_child_map = []
    self.left_mask = []
    self.right_mask = []

    for tree in sent.trees:
      left_child_map, left_mask = self.build_child_map(tree, 0)
      right_child_map, right_mask = self.build_child_map(tree, 1)

      self.left_child_map.append(left_child_map)
      self.right_child_map.append(right_child_map)
      self.left_mask.append(left_mask)
      self.right_mask.append(right_mask)

    assert len(self.left_child_map) == sent.batch_size()
    assert len(self.left_child_map) == len(self.right_child_map)
    assert len(self.left_child_map) == len(self.left_mask)
    assert len(self.left_child_map) == len(self.right_mask)
    for i in range(len(self.left_child_map)):
      assert len(self.left_child_map[i]) == len(self.right_child_map[i])
      assert len(self.left_child_map[i]) == len(self.left_mask[i])
      assert len(self.left_child_map[i]) == len(self.right_mask[i])

    longest = len(max(self.left_child_map, key=len))
    for i in range(len(self.left_child_map)):
      diff = longest - len(self.left_child_map[i])
      self.left_child_map[i] += [0] * diff
      self.left_mask[i] += [0] * diff
      self.right_child_map[i] += [0] * diff
      self.right_mask[i] += [0] * diff


    self.left_mask = dy.inputTensor(self.transpose(self.left_mask), batched=True)
    self.right_mask = dy.inputTensor(self.transpose(self.right_mask), batched=True)
    return super().init_sent(encoded, sent)

  def select_from_state(self, state, child_map, mask):
    r = []
    for i, row in enumerate(child_map):
      selected = dy.select_cols(dy.pick_batch_elem(state, i), row)
      r.append(selected)
    r = dy.concatenate_to_batch(r)
    r = dy.cmult(r, mask)
    return selected

  def update(self, dec_state, att_state: AttenderState, attention: dy.Expression):
    dec_state_b = dy.concatenate_cols([dec_state for _ in range(self.I.dim()[0][1])])

    state = dy.concatenate([att_state, dy.transpose(attention)])
    left_state = self.select_from_state(state, self.left_child_map, dy.transpose(self.left_mask))
    right_state = self.select_from_state(state, self.right_child_map, dy.transpose(self.right_mask))

    h_in = dy.concatenate([self.I, dy.transpose(attention), dec_state_b, left_state, right_state])
    h_out = self.coverage_updater.add_input_to_prev(att_state, h_in)[0]
    assert h_out.dim() == att_state.dim()
    return h_out

class DotAttender(Attender, Serializable):
  """
  Implements dot product attention of https://arxiv.org/abs/1508.04025
  Also (optionally) perform scaling of https://arxiv.org/abs/1706.03762

  Args:
    scale: whether to perform scaling
    truncate_dec_batches: currently unsupported
  """

  yaml_tag = '!DotAttender'

  @serializable_init
  def __init__(self,
               scale: bool = True,
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    if truncate_dec_batches: raise NotImplementedError("truncate_dec_batches not yet implemented for DotAttender")
    self.curr_sent = None
    self.scale = scale
    self.attention_vecs = []

  def init_sent(self, encoded: expression_seqs.ExpressionSequence, sent) -> None:
    self.curr_sent = encoded
    self.attention_vecs = []
    self.I = dy.transpose(self.curr_sent.as_tensor())
    return None

  def calc_attention(self, dec_state: dy.Expression, att_state: AttenderState = None) -> dy.Expression:
    scores = self.I * dec_state
    if self.scale:
      scores /= math.sqrt(dec_state.dim()[0][0])
    if self.curr_sent.mask is not None:
      scores = self.curr_sent.mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

class BilinearAttender(Attender, Serializable):
  """
  Implements a bilinear attention, equivalent to the 'general' linear
  attention of https://arxiv.org/abs/1508.04025

  Args:
    input_dim: input dimension; if None, use exp_global.default_layer_dim
    state_dim: dimension of dec_state inputs; if None, use exp_global.default_layer_dim
    param_init: how to initialize weight matrices; if None, use ``exp_global.param_init``
    truncate_dec_batches: currently unsupported
  """

  yaml_tag = '!BilinearAttender'

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               state_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    if truncate_dec_batches: raise NotImplementedError("truncate_dec_batches not yet implemented for BilinearAttender")
    self.input_dim = input_dim
    self.state_dim = state_dim
    param_collection = param_collections.ParamManager.my_params(self)
    self.Wa = param_collection.add_parameters((input_dim, state_dim), init=param_init.initializer((input_dim, state_dim)))
    self.curr_sent = None

  def init_sent(self, encoded: expression_seqs.ExpressionSequence, sent) -> None:
    self.curr_sent = encoded
    self.attention_vecs = []
    self.I = self.curr_sent.as_tensor()
    return None

  # TODO(philip30): Please apply masking here
  def calc_attention(self, dec_state: dy.Expression, att_state: AttenderState = None) -> dy.Expression:
    logger.warning("BilinearAttender does currently not do masking, which may harm training results.")
    scores = (dy.transpose(dec_state) * self.Wa) * self.I
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return dy.transpose(normalized)

class LatticeBiasedMlpAttender(MlpAttender, Serializable):
  """
  Modified MLP attention, where lattices are assumed as input and the attention is biased toward confident nodes.

  Args:
    input_dim: input dimension
    state_dim: dimension of dec_state inputs
    hidden_dim: hidden MLP dimension
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
  """

  yaml_tag = '!LatticeBiasedMlpAttender'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               state_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    super().__init__(input_dim=input_dim, state_dim=state_dim, hidden_dim=hidden_dim, param_init=param_init,
                     bias_init=bias_init, truncate_dec_batches=truncate_dec_batches)

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self.cur_sent_bias = np.full((src.sent_len(), 1, src.batch_size()), -1e10)
    for batch_i, lattice_batch_elem in enumerate(src):
      for node_i, node in enumerate(lattice_batch_elem.nodes):
        self.cur_sent_bias[node_i, 0, batch_i] = node.marginal_log_prob
    self.cur_sent_bias_expr = None

  def calc_attention(self, dec_state: dy.Expression, att_state: AttenderState = None) -> dy.Expression:
    WI = self.WI
    curr_sent_mask = self.curr_sent.mask
    if self.truncate_dec_batches:
      if curr_sent_mask:
        dec_state, WI, curr_sent_mask = batchers.truncate_batches(dec_state, WI, curr_sent_mask)
      else:
        dec_state, WI = batchers.truncate_batches(dec_state, WI)
    h = dy.tanh(dy.colwise_add(WI, self.V * dec_state))
    scores = dy.transpose(self.U * h)
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator = -1e10)
    if self.cur_sent_bias_expr is None: self.cur_sent_bias_expr = dy.inputTensor(self.cur_sent_bias, batched=True)
    normalized = dy.softmax(scores + self.cur_sent_bias_expr)
    self.attention_vecs.append(normalized)
    return normalized

