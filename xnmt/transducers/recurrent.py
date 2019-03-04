import numbers
import collections.abc
from typing import List, Optional, Sequence, Tuple, Union

import sys
import numpy as np
import dynet as dy


from xnmt import expression_seqs, param_collections, param_initializers
from xnmt.modelparts import transforms, attenders
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.transducers import base as transducers
from xnmt.persistence import bare, Ref, Serializable, serializable_init, Path

class UniLSTMState(object):
  """
  State object for UniLSTMSeqTransducer.
  """
  def __init__(self,
               network: 'UniLSTMSeqTransducer',
               prev: Optional['UniLSTMState'] = None,
               c: Sequence[dy.Expression] = None,
               h: Sequence[dy.Expression] = None) -> None:
    self._network = network
    if c is None:
      c = [dy.zeroes(dim=(network.hidden_dim,)) for _ in range(network.num_layers)]
    if h is None:
      h = [dy.zeroes(dim=(network.hidden_dim,)) for _ in range(network.num_layers)]
    self._c = tuple(c)
    self._h = tuple(h)
    self._prev = prev

  def add_input(self, x: Union[dy.Expression, Sequence[dy.Expression]]):
    new_c, new_h = self._network.add_input_to_prev(self, x)
    return UniLSTMState(self._network, prev=self, c=new_c, h=new_h)

  def b(self) -> 'UniLSTMSeqTransducer':
    return self._network

  def h(self) -> Sequence[dy.Expression]:
    return self._h

  def s(self) -> Sequence[dy.Expression]:
    return self._c + self._h

  def prev(self) -> 'UniLSTMState':
    return self._prev

  def set_h(self, es: Optional[Sequence[dy.Expression]] = None) -> 'UniLSTMState':
    if es is not None:
      assert len(es) == self._network.num_layers
    self._h = tuple(es)
    return self

  def set_s(self, es: Optional[Sequence[dy.Expression]] = None) -> 'UniLSTMState':
    if es is not None:
      assert len(es) == 2 * self._network.num_layers
    self._c = tuple(es[:self._network.num_layers])
    self._h = tuple(es[self._network.num_layers:])
    return self

  def output(self) -> dy.Expression:
    return self._h[-1]

  def __getitem__(self, item):
    return UniLSTMState(network=self._network,
                        prev=self._prev,
                        c=[ci[item] for ci in self._c],
                        h=[hi[item] for hi in self._h])

class UniGRUSeqTransducer(transducers.SeqTransducer, Serializable):
  yaml_tag = '!UniGRUSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    model = param_collections.ParamManager.my_params(self)
    W_dim = (self.hidden_dim, self.input_dim)
    U_dim = (self.hidden_dim, self.hidden_dim)
    b_dim = (self.hidden_dim,)
    h_dim = (self.hidden_dim,)

    self.Wz = model.add_parameters(W_dim, init=param_init.initializer(W_dim))
    self.Wr = model.add_parameters(W_dim, init=param_init.initializer(W_dim))
    self.Wu = model.add_parameters(W_dim, init=param_init.initializer(W_dim))

    self.Uz = model.add_parameters(U_dim, init=param_init.initializer(U_dim))
    self.Ur = model.add_parameters(U_dim, init=param_init.initializer(U_dim))
    self.Uu = model.add_parameters(U_dim, init=param_init.initializer(U_dim))

    self.bz = model.add_parameters(b_dim, init=bias_init.initializer(b_dim))
    self.br = model.add_parameters(b_dim, init=bias_init.initializer(b_dim))
    self.bu = model.add_parameters(b_dim, init=bias_init.initializer(b_dim))

    self.initial_h = model.add_parameters(h_dim, init=param_init.initializer(h_dim))

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def initial_state(self) -> dy.Expression:
    return self.initial_h

  def add_input_to_prev(self, prev_state: dy.Expression, x: dy.Expression) -> dy.Expression:
    z = dy.logistic(self.Wz * x + self.Uz * prev_state + self.bz)
    r = dy.logistic(self.Wr * x + self.Ur * prev_state + self.br)
    u = dy.tanh(self.Wu * x + self.Uu * prev_state + self.bu)
    h = dy.cmult((1 - z), prev_state) + dy.cmult(z, u)
    return h

class UniLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This implements a single LSTM layer based on the memory-friendly dedicated DyNet nodes.
  It works similar to DyNet's CompactVanillaLSTMBuilder, but in addition supports
  taking multiple inputs that are concatenated on-the-fly.

  Args:
    layers (int): number of layers
    input_dim (int): input dimension
    hidden_dim (int): hidden dimension
    dropout (float): dropout probability
    weightnoise_std (float): weight noise standard deviation
    param_init (ParamInitializer): how to initialize weight matrices
    bias_init (ParamInitializer): how to initialize bias vectors
    yaml_path (str):
    decoder_input_dim (int): input dimension of the decoder; if ``yaml_path`` contains 'decoder' and ``decoder_input_feeding`` is True, this will be added to ``input_dim``
    decoder_input_feeding (bool): whether this transducer is part of an input-feeding decoder; cf. ``decoder_input_dim``
  """
  yaml_tag = '!UniLSTMSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               weightnoise_std: numbers.Real = Ref("exp_global.weight_noise", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               yaml_path: Path = Path(),
               decoder_input_dim: Optional[numbers.Integral] = Ref("exp_global.default_layer_dim", default=None),
               decoder_input_feeding: bool = True) -> None:
    self.num_layers = layers
    model = param_collections.ParamManager.my_params(self)
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout
    self.weightnoise_std = weightnoise_std
    self.input_dim = input_dim
    self.total_input_dim = input_dim
    if yaml_path is not None and "decoder" in yaml_path:
      if decoder_input_feeding:
        self.total_input_dim += decoder_input_dim

    if not isinstance(param_init, collections.abc.Sequence):
      param_init = [param_init] * layers
    if not isinstance(bias_init, collections.abc.Sequence):
        bias_init = [bias_init] * layers

    # [i; f; o; g]
    self.p_Wx = [model.add_parameters(dim=(hidden_dim*4, self.total_input_dim), init=param_init[0].initializer((hidden_dim*4, self.total_input_dim), num_shared=4))]
    self.p_Wx += [model.add_parameters(dim=(hidden_dim*4, hidden_dim), init=param_init[i].initializer((hidden_dim*4, hidden_dim), num_shared=4)) for i in range(1, layers)]
    self.p_Wh = [model.add_parameters(dim=(hidden_dim*4, hidden_dim), init=param_init[i].initializer((hidden_dim*4, hidden_dim), num_shared=4)) for i in range(layers)]
    self.p_b  = [model.add_parameters(dim=(hidden_dim*4,), init=bias_init[i].initializer((hidden_dim*4,), num_shared=4)) for i in range(layers)]

    self.dropout_mask_x = None
    self.dropout_mask_h = None

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None
    self.Wx = [dy.parameter(Wx) for Wx in self.p_Wx]
    self.Wh = [dy.parameter(Wh) for Wh in self.p_Wh]
    self.b = [dy.parameter(b) for b in self.p_b]
    self.dropout_mask_x = None
    self.dropout_mask_h = None

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def initial_state(self) -> UniLSTMState:
    return UniLSTMState(self)

  def set_dropout(self, dropout: numbers.Real) -> None:
    self.dropout_rate = dropout

  def set_dropout_masks(self, batch_size: numbers.Integral = 1) -> None:
    if self.dropout_rate > 0.0 and self.train:
      retention_rate = 1.0 - self.dropout_rate
      scale = 1.0 / retention_rate
      self.dropout_mask_x = [dy.random_bernoulli((self.total_input_dim,), retention_rate, scale, batch_size=batch_size)]
      self.dropout_mask_x += [dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size) for _ in range(1, self.num_layers)]
      self.dropout_mask_h = [dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size) for _ in range(self.num_layers)]

  def add_input_to_prev(self, prev_state: UniLSTMState, x: Union[dy.Expression, Sequence[dy.Expression]]) \
          -> Tuple[Sequence[dy.Expression]]:
    if isinstance(x, dy.Expression):
      x = [x]
    elif type(x) != list:
      x = list(x)

    if self.dropout_rate > 0.0 and self.train and self.dropout_mask_x is None:
      self.set_dropout_masks()

    new_c, new_h = [], []
    for layer_i in range(self.num_layers):
      if self.dropout_rate > 0.0 and self.train:
        # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
        gates = dy.vanilla_lstm_gates_dropout_concat(
          x, prev_state._h[layer_i], self.Wx[layer_i], self.Wh[layer_i], self.b[layer_i],
          self.dropout_mask_x[layer_i], self.dropout_mask_h[layer_i],
          self.weightnoise_std if self.train else 0.0)
      else:
        gates = dy.vanilla_lstm_gates_concat(
          x, prev_state._h[layer_i], self.Wx[layer_i], self.Wh[layer_i], self.b[layer_i],
          self.weightnoise_std if self.train else 0.0)
      new_c.append(dy.vanilla_lstm_c(prev_state._c[layer_i], gates))
      new_h.append(dy.vanilla_lstm_h(new_c[-1], gates))
      x = [new_h[-1]]

    return new_c, new_h

  def transduce(self, expr_seq: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':
    """
    transduce the sequence, applying masks if given (masked timesteps simply copy previous h / c)

    Args:
      expr_seq: expression sequence or list of expression sequences (where each inner list will be concatenated)
    Returns:
      expression sequence
    """
    if isinstance(expr_seq, expression_seqs.ExpressionSequence):
      expr_seq = [expr_seq]
    batch_size = expr_seq[0][0].dim()[1]
    seq_len = len(expr_seq[0])

    if self.dropout_rate > 0.0 and self.train:
      self.set_dropout_masks(batch_size=batch_size)

    cur_input = expr_seq
    self._final_states = []
    for layer_i in range(self.num_layers):
      h = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
      c = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
      for pos_i in range(seq_len):
        x_t = [cur_input[j][pos_i] for j in range(len(cur_input))]
        if isinstance(x_t, dy.Expression):
          x_t = [x_t]
        elif type(x_t) != list:
          x_t = list(x_t)
        if sum([x_t_i.dim()[0][0] for x_t_i in x_t]) != self.total_input_dim:
          found_dim = sum([x_t_i.dim()[0][0] for x_t_i in x_t])
          raise ValueError(f"VanillaLSTMGates: x_t has inconsistent dimension {found_dim}, expecting {self.total_input_dim}")
        if self.dropout_rate > 0.0 and self.train:
          # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
          gates_t = dy.vanilla_lstm_gates_dropout_concat(x_t,
                                                         h[-1],
                                                         self.Wx[layer_i],
                                                         self.Wh[layer_i],
                                                         self.b[layer_i],
                                                         self.dropout_mask_x[layer_i],
                                                         self.dropout_mask_h[layer_i],
                                                         self.weightnoise_std if self.train else 0.0)
        else:
          gates_t = dy.vanilla_lstm_gates_concat(x_t, h[-1], self.Wx[layer_i], self.Wh[layer_i], self.b[layer_i], self.weightnoise_std if self.train else 0.0)
        c_t = dy.vanilla_lstm_c(c[-1], gates_t)
        h_t = dy.vanilla_lstm_h(c_t, gates_t)
        if expr_seq[0].mask is None or np.isclose(np.sum(expr_seq[0].mask.np_arr[:,pos_i:pos_i+1]), 0.0):
          c.append(c_t)
          h.append(h_t)
        else:
          c.append(expr_seq[0].mask.cmult_by_timestep_expr(c_t,pos_i,True) + expr_seq[0].mask.cmult_by_timestep_expr(c[-1],pos_i,False))
          h.append(expr_seq[0].mask.cmult_by_timestep_expr(h_t,pos_i,True) + expr_seq[0].mask.cmult_by_timestep_expr(h[-1],pos_i,False))
      self._final_states.append(transducers.FinalTransducerState(h[-1], c[-1]))
      cur_input = [h[1:]]

    return expression_seqs.ExpressionSequence(expr_list=h[1:], mask=expr_seq[0].mask)

class BiLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This implements a bidirectional LSTM and requires about 8.5% less memory per timestep
  than DyNet's CompactVanillaLSTMBuilder due to avoiding concat operations.
  It uses 2 :class:`xnmt.lstm.UniLSTMSeqTransducer` objects in each layer.

  Args:
    layers (int): number of layers
    input_dim (int): input dimension
    hidden_dim (int): hidden dimension
    dropout (float): dropout probability
    weightnoise_std (float): weight noise standard deviation
    param_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects
                specifying how to initialize weight matrices. If a list is given, each entry denotes one layer.
    bias_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects
               specifying how to initialize bias vectors. If a list is given, each entry denotes one layer.
    forward_layers: set automatically
    backward_layers: set automatically
  """
  yaml_tag = '!BiLSTMSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               weightnoise_std: numbers.Real = Ref("exp_global.weight_noise", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               forward_layers : Optional[Sequence[UniLSTMSeqTransducer]] = None,
               backward_layers: Optional[Sequence[UniLSTMSeqTransducer]] = None) -> None:
    self.num_layers = layers
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout
    self.weightnoise_std = weightnoise_std
    assert hidden_dim % 2 == 0
    self.forward_layers = self.add_serializable_component("forward_layers", forward_layers, lambda: [
      UniLSTMSeqTransducer(input_dim=input_dim if i == 0 else hidden_dim, hidden_dim=hidden_dim // 2, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init[i] if isinstance(param_init, collections.abc.Sequence) else param_init,
                           bias_init=bias_init[i] if isinstance(bias_init, collections.abc.Sequence) else bias_init) for i in
      range(layers)])
    self.backward_layers = self.add_serializable_component("backward_layers", backward_layers, lambda: [
      UniLSTMSeqTransducer(input_dim=input_dim if i == 0 else hidden_dim, hidden_dim=hidden_dim // 2, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init[i] if isinstance(param_init, collections.abc.Sequence) else param_init,
                           bias_init=bias_init[i] if isinstance(bias_init, collections.abc.Sequence) else bias_init) for i in
      range(layers)])

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def transduce(self, es: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':
    mask = es.mask
     # first layer
    forward_es = self.forward_layers[0].transduce(es)
    rev_backward_es = self.backward_layers[0].transduce(expression_seqs.ReversedExpressionSequence(es))

    for layer_i in range(1, len(self.forward_layers)):
      new_forward_es = self.forward_layers[layer_i].transduce([forward_es, expression_seqs.ReversedExpressionSequence(rev_backward_es)])
      rev_backward_es = expression_seqs.ExpressionSequence(
        self.backward_layers[layer_i].transduce([expression_seqs.ReversedExpressionSequence(forward_es), rev_backward_es]).as_list(),
        mask=mask)
      forward_es = new_forward_es

    self._final_states = [
      transducers.FinalTransducerState(dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].main_expr()]),
                                       dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].cell_expr()])) \
      for layer_i in range(len(self.forward_layers))]
    return expression_seqs.ExpressionSequence(expr_list=[dy.concatenate([forward_es[i],rev_backward_es[-i-1]]) for i in range(len(forward_es))], mask=mask)


class CustomLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This implements an LSTM builder based on elementary DyNet operations.
  It is more memory-hungry than the compact LSTM, but can be extended more easily.
  It currently does not support dropout or multiple layers and is mostly meant as a
  starting point for LSTM extensions.

  Args:
    layers (int): number of layers
    input_dim (int): input dimension; if None, use exp_global.default_layer_dim
    hidden_dim (int): hidden dimension; if None, use exp_global.default_layer_dim
    param_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects
                specifying how to initialize weight matrices. If a list is given, each entry denotes one layer.
                If None, use ``exp_global.param_init``
    bias_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects
               specifying how to initialize bias vectors. If a list is given, each entry denotes one layer.
               If None, use ``exp_global.param_init``
  """
  yaml_tag = "!CustomLSTMSeqTransducer"

  @serializable_init
  def __init__(self,
               layers: numbers.Integral,
               input_dim: numbers.Integral,
               hidden_dim: numbers.Integral,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    if layers!=1: raise RuntimeError("CustomLSTMSeqTransducer supports only exactly one layer")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    model = param_collections.ParamManager.my_params(self)

    # [i; f; o; g]
    self.p_Wx = model.add_parameters(dim=(hidden_dim*4, input_dim), init=param_init.initializer((hidden_dim*4, input_dim)))
    self.p_Wh = model.add_parameters(dim=(hidden_dim*4, hidden_dim), init=param_init.initializer((hidden_dim*4, hidden_dim)))
    self.p_b  = model.add_parameters(dim=(hidden_dim*4,), init=bias_init.initializer((hidden_dim*4,)))

  def transduce(self, xs: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':
    Wx = dy.parameter(self.p_Wx)
    Wh = dy.parameter(self.p_Wh)
    b = dy.parameter(self.p_b)
    h = []
    c = []
    for i, x_t in enumerate(xs):
      if i==0:
        tmp = dy.affine_transform([b, Wx, x_t])
      else:
        tmp = dy.affine_transform([b, Wx, x_t, Wh, h[-1]])
      i_ait = dy.pick_range(tmp, 0, self.hidden_dim)
      i_aft = dy.pick_range(tmp, self.hidden_dim, self.hidden_dim*2)
      i_aot = dy.pick_range(tmp, self.hidden_dim*2, self.hidden_dim*3)
      i_agt = dy.pick_range(tmp, self.hidden_dim*3, self.hidden_dim*4)
      i_it = dy.logistic(i_ait)
      i_ft = dy.logistic(i_aft + 1.0)
      i_ot = dy.logistic(i_aot)
      i_gt = dy.tanh(i_agt)
      if i==0:
        c.append(dy.cmult(i_it, i_gt))
      else:
        c.append(dy.cmult(i_ft, c[-1]) + dy.cmult(i_it, i_gt))
      h.append(dy.cmult(i_ot, dy.tanh(c[-1])))
    return h

from xnmt.sent import SyntaxTree
def linearize(tree: SyntaxTree):
  """Converts a SyntaxTree of vectors into a linear sequence of vectors"""
  r = [tree.label]
  for child in tree.children:
    r += linearize(child)
  return r

def zip_trees(trees):
  """Takes a list of trees, all with the same structure,
  and returns a new tree wherein each node's vector is the
  concatenation of the same node's vectors in the input trees"""
  # Verify that all the trees have the same structure
  assert len(trees) > 0
  for i in range(1, len(trees)):
    assert len(trees[i].children) == len(trees[0].children)

  children = [zip_trees([tree.children[i] for tree in trees]) for i in range(len(trees[0].children))]
  #label = dy.concatenate([tree.label for tree in trees])
  label = dy.esum([tree.label for tree in trees])
  return SyntaxTree(label, children)


class SyntaxTreeEncoder(transducers.SeqTransducer, Serializable):
  """
  Args:
    layers (int): number of layers
    input_dim (int): input dimension
    hidden_dim (int): hidden dimension
    dropout (float): dropout probability
    weightnoise_std (float): weight noise standard deviation
    param_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects
                specifying how to initialize weight matrices. If a list is given, each entry denotes one layer.
    bias_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects
               specifying how to initialize bias vectors. If a list is given, each entry denotes one layer.
  """
  yaml_tag = '!SyntaxTreeEncoder'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers=1,
               input_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               dropout=Ref("exp_global.dropout", default=0.0),
               weightnoise_std=Ref("exp_global.weight_noise", default=0.0),
               param_init=Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               transform=bare(transforms.Linear, bias=False),
               root_main_transform=bare(transforms.NonLinear),
               root_cell_transform=bare(transforms.NonLinear),
               inside_fwd_layers=None, inside_rev_layers=None,
               outside_left_layers=None, outside_right_layers=None, mlps=None):
    self.num_layers = layers
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout
    self.weightnoise_std = weightnoise_std
    self.transform = transform
    self.root_main_transform = root_main_transform
    self.root_cell_transform = root_cell_transform

    self.inside_fwd_layers = self.add_serializable_component("inside_fwd_layers", inside_fwd_layers, lambda: [
      UniLSTMSeqTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init[i] if isinstance(param_init, Sequence) else param_init,
                           bias_init=bias_init[i] if isinstance(bias_init, Sequence) else bias_init) for i in
      range(layers)])

    self.inside_rev_layers = self.add_serializable_component("inside_rev_layers", inside_rev_layers, lambda: [
      UniLSTMSeqTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init[i] if isinstance(param_init, Sequence) else param_init,
                           bias_init=bias_init[i] if isinstance(bias_init, Sequence) else bias_init) for i in
      range(layers)])

    self.outside_left_layers = self.add_serializable_component("outside_left_layers", outside_left_layers, lambda: [
      UniLSTMSeqTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init[i] if isinstance(param_init, Sequence) else param_init,
                           bias_init=bias_init[i] if isinstance(bias_init, Sequence) else bias_init) for i in
      range(layers)])

    self.outside_right_layers = self.add_serializable_component("outside_right_layers", outside_right_layers, lambda: [
      UniLSTMSeqTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init[i] if isinstance(param_init, Sequence) else param_init,
                           bias_init=bias_init[i] if isinstance(bias_init, Sequence) else bias_init) for i in
      range(layers)])

    self.mlps = self.add_serializable_component('mlps', mlps, lambda: [transforms.MLP(input_dim=3*hidden_dim,
                                                hidden_dim=hidden_dim, output_dim=hidden_dim,
                                                #dropout=dropout, # TODO: Why doesn't MLP support dropout?
                                                param_init=param_init[i] if isinstance(param_init, Sequence) else param_init,
                                                bias_init=bias_init[i] if isinstance(bias_init, Sequence) else bias_init) for i in range(layers)])

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.root_emb = None

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    main = self.root_main_transform.transform(self.root_emb)
    cell = self.root_cell_transform.transform(self.root_emb)
    return [transducers.FinalTransducerState(main, cell)]

  def embed_subtree_inside(self, tree: SyntaxTree, layer_idx=0):
    if len(tree.children) == 0:
      return SyntaxTree(tree.label, tree.children)
    else:
      children = [self.embed_subtree_inside(child, layer_idx)
                    for child in tree.children]
      child_embs = [child.label for child in children]
      fwd_expr_seq = expression_seqs.ExpressionSequence([tree.label] + child_embs)
      rev_expr_seq = expression_seqs.ExpressionSequence([tree.label] + child_embs[::-1])
      fwd_summary = self.inside_fwd_layers[layer_idx].transduce(fwd_expr_seq)[-1]
      rev_summary = self.inside_rev_layers[layer_idx].transduce(rev_expr_seq)[-1]
      summary = fwd_summary + rev_summary
      return SyntaxTree(summary, children)

  def embed_subtree_outside(self, tree: SyntaxTree,
                              left_sibs: List[dy.Expression],
                              right_sibs: List[dy.Expression],
                              parent_outside: dy.Expression,
                              layer_idx=0):
    # Should combine (summary vectors of):
    # 1) Left siblings' inside representations
    # 2) Right siblings' inside representations
    # 3) Parent's outside representation
    zeros = dy.zeros(self.hidden_dim)
    if len(left_sibs) > 0:
      left_sibs = expression_seqs.ExpressionSequence(left_sibs)
      left_context = self.outside_left_layers[layer_idx].transduce(left_sibs)[-1]
    else:
      left_context = zeros
    if len(right_sibs) > 0:
      right_sibs = expression_seqs.ExpressionSequence(right_sibs[::-1])
      right_context = self.outside_right_layers[layer_idx].transduce(right_sibs)[-1]
    else:
      right_context = zeros
    summary_in = dy.concatenate([left_context, right_context, parent_outside])
    summary = self.mlps[layer_idx].transform(summary_in)

    new_children = []
    for i, child in enumerate(tree.children):
      left_sibs = [node.label for node in tree.children[:i]]
      right_sibs = [node.label for node in tree.children[i+1:]]
      new_child = self.embed_subtree_outside(
        child, left_sibs, right_sibs, summary, layer_idx)
      new_children.append(new_child)
    return SyntaxTree(summary, new_children)

  def transform_labels(self, tree: SyntaxTree):
    new_label = self.transform.transform(tree.label)
    children = [self.transform_labels(child) for child in tree.children]
    return SyntaxTree(new_label, children)

  def embed_tree(self, tree: SyntaxTree):
    # TODO: Better computation of root outside
    root_outside = dy.zeros(self.hidden_dim)
    for i in range(self.layers):
      if i == 0 and self.input_dim != self.hidden_dim:
        tree = self.transform_labels(tree)
      inside_tree = self.embed_subtree_inside(tree, i)
      outside_tree = self.embed_subtree_outside(inside_tree, [], [], root_outside, i)
      tree = zip_trees([inside_tree, outside_tree, tree])
    return tree

  def transduce(self, trees: SyntaxTree) -> 'expression_seqs.ExpressionSequence':
    if type(trees) != list:
      tree = self.embed_tree(trees)
      self.root_emb = tree.label
      return linearize(tree)
    else:
      assert len(trees) == 1
      output = [self.transduce(t) for t in trees]
      assert len(output) == 1
      output = output[0] # XXX
      return expression_seqs.ExpressionSequence(output)

  def shared_params(self):
    return [{".input_dim", ".transform.input_dim"},
            {".hidden_dim", ".transform.output_dim"},
            {".hidden_dim", ".root_main_transform.input_dim"},
            {".hidden_dim", ".root_main_transform.output_dim"},
            {".hidden_dim", ".root_cell_transform.input_dim"},
            {".hidden_dim", ".root_cell_transform.output_dim"}]

class BinarySyntaxTreeEncoder(transducers.SeqTransducer, Serializable):
  yaml_tag = '!BinarySyntaxTreeEncoder'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers=1,
               input_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               dropout=Ref("exp_global.dropout", default=0.0),
               emb_transform=bare(transforms.Linear, bias=False),
               root_transform=bare(transforms.NonLinear),
               attender=bare(attenders.MlpAttender),
               mlp=None):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout
    self.emb_transform = emb_transform
    self.root_transform = root_transform
    self.attender = attender
    self.mlp = self.add_serializable_component('mlp', mlp, lambda: transforms.MLP(input_dim=2*hidden_dim,
                                               hidden_dim=hidden_dim, output_dim=hidden_dim))
  @handle_xnmt_event
  def on_start_sent(self, src):
    pass

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    # TODO: Real final states
    z = dy.zeros(self.hidden_dim)
    return [transducers.FinalTransducerState(z, z)]

  def transform_labels(self, tree: SyntaxTree):
    new_label = self.emb_transform.transform(tree.label)
    children = [self.transform_labels(child) for child in tree.children]
    return SyntaxTree(new_label, children)

  def embed_subtree_inside(self, tree: SyntaxTree):
    if len(tree.children) == 0:
      return SyntaxTree(tree.label, tree.children)
    elif len(tree.children) == 1:
      child = self.embed_subtree_inside(tree.children[0])
      new_label = tree.label + child.label
      return SyntaxTree(new_label, [child])
    elif len(tree.children) == 2:
      children = [self.embed_subtree_inside(child) for child in tree.children]
      child_seq = expression_seqs.ExpressionSequence([child.label for child in children])
      self.attender.init_sent(child_seq)
      new_label = tree.label + self.attender.calc_context(tree.label)
      new_label = dy.reshape(new_label, (self.hidden_dim,))
      return SyntaxTree(new_label, children)
    else:
      assert False, 'Invalid arity for BinarySyntaxTreeEncoder %d' % len(tree.children)

  def embed_subtree_outside(self, tree: SyntaxTree, parent_outside):
    outside = self.mlp.transform(dy.concatenate([parent_outside, tree.label]))
    # TODO should we add outside here, or just leave the root node's
    # embedding as just its inside embedding?
    new_label = tree.label + outside
    children = [self.embed_subtree_outside(child, outside) for child in tree.children]
    return SyntaxTree(new_label, children)

  def embed_tree_outside(self, tree: SyntaxTree):
    root_outside = self.root_transform.transform(tree.label)
    new_label = tree.label + root_outside
    children = [self.embed_subtree_outside(child, root_outside) for child in tree.children]
    return SyntaxTree(new_label, children)

  def embed_tree(self, tree: SyntaxTree):
    tree = self.transform_labels(tree)
    inside_tree = self.embed_subtree_inside(tree)
    outside_tree = self.embed_tree_outside(inside_tree)
    return outside_tree

  def transduce(self, trees: SyntaxTree) -> expression_seqs.ExpressionSequence:
    if type(trees) != list:
      tree = self.embed_tree(trees)
      return linearize(tree)
    else:
      assert len(trees) == 1
      output = [self.transduce(t) for t in trees]
      assert len(output) == 1
      output = output[0] # XXX
      return expression_seqs.ExpressionSequence(output)
    pass

  def shared_params(self):
    return [{".hidden_dim", ".root_transform.input_dim"},
            {".hidden_dim", ".root_transform.output_dim"},
            {".input_dim", ".emb_transform.input_dim"},
            {".hidden_dim", ".emb_transform.output_dim"},
            {".hidden_dim", ".attender.input_dim"},
            {".hidden_dim", ".attender.state_dim"},
            {".hidden_dim", ".attender.hidden_dim"}]

class TreeRNN(transducers.SeqTransducer):
  """yaml_tag='!TreeRNN'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.create_parameters(param_init, bias_init)"""

  @handle_xnmt_event
  def on_start_sent(self, src):
    pass

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    # TODO: Real final states
    z = dy.zeros(self.hidden_dim)
    return [transducers.FinalTransducerState(z, z)]

  def compute_gate(self, key, layer_idx, x, children, activation=dy.logistic):
    W = getattr(self, 'W%s' % key)[layer_idx]
    b = getattr(self, 'b%s' % key)[layer_idx]
    r = W * x + b

    for i, child in enumerate(children):
      U = getattr(self, 'U%s%d' % (key, i))[layer_idx]
      r += U * child.h

    return activation(r)

  def transduce(self, trees: SyntaxTree) -> expression_seqs.ExpressionSequence:
    if type(trees) != list:
      for layer_idx in range(self.layers):
        trees = self.embed_tree(trees, layer_idx)
      return linearize(trees)
    else:
      assert len(trees) == 1
      output = [self.transduce(t) for t in trees]
      assert len(output) == 1
      output = output[0] # XXX
      return expression_seqs.ExpressionSequence(output)

  def create_parameter(self, model, name, dim, init):
    if not hasattr(self, name):
      setattr(self, name, [])
    param_list = getattr(self, name)
    assert type(param_list) == list
    param = model.add_parameters(dim, init=init.initializer(dim))
    param_list.append(param)

  def shared_params(self):
    return []

class TreeLSTM(TreeRNN, Serializable):
  yaml_tag = '!TreeLSTM'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    self.layers = layers
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.create_parameters(layers, param_init, bias_init)

  def create_parameters(self, layers, param_init, bias_init):
    model = param_collections.ParamManager.my_params(self)
    for layer_idx in range(layers):
      W_dim = (self.hidden_dim, self.input_dim) if layer_idx == 0 else (self.hidden_dim, self.hidden_dim)
      U_dim = (self.hidden_dim, self.hidden_dim)
      b_dim = (self.hidden_dim,)

      for gate in ['i', 'f0', 'f1', 'o', 'u']:
        self.create_parameter(model, 'W' + gate, W_dim, param_init)
        self.create_parameter(model, 'b' + gate, b_dim, bias_init)
        for child_idx in ['0', '1']:
          self.create_parameter(model, 'U' + gate + child_idx, U_dim, param_init)

  def embed_tree(self, tree: SyntaxTree, layer_idx):
    assert len(tree.children) <= 2
    children = [self.embed_tree(child, layer_idx) for child in tree.children]
    h, c = self.compute_output(layer_idx, tree.label, children)
    new_tree = SyntaxTree(h, children)
    new_tree.h = h
    new_tree.c = c
    return new_tree

  def compute_output(self, layer_idx, label, children):
    i = self.compute_gate('i', layer_idx,  label, children)
    f = [self.compute_gate('f%d' % j, layer_idx, label, children) for j in range(len(children))]
    o = self.compute_gate('o', layer_idx, label, children)
    u = self.compute_gate('u', layer_idx, label, children, activation=dy.tanh)
    c = dy.cmult(i, u)
    for j, child in enumerate(children):
      c += dy.cmult(f[j], child.c)
    h = dy.cmult(o, dy.tanh(c))
    return h, c

class TreeGRU(TreeRNN, Serializable):
  yaml_tag = '!TreeGRU'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    self.layers = layers
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.create_parameters(layers, param_init, bias_init)

  def create_parameters(self, layers, param_init, bias_init):
    model = param_collections.ParamManager.my_params(self)
    for layer_idx in range(layers):
      W_dim = (self.hidden_dim, self.input_dim) if layer_idx == 0 else (self.hidden_dim, self.hidden_dim)
      U_dim = (self.hidden_dim, self.hidden_dim)
      b_dim = (self.hidden_dim,)

      for gate in ['i', 'f0', 'f1', 'r0', 'r1', 'u']:
        self.create_parameter(model, 'W' + gate, W_dim, param_init)
        self.create_parameter(model, 'b' + gate, b_dim, bias_init)
        for child_idx in ['0', '1']:
          self.create_parameter(model,  'U' + gate + child_idx, U_dim, param_init)


  def compute_u(self, layer_idx, x, children, r):
    key = 'u'
    W = getattr(self, 'W%s' % key)[layer_idx]
    b = getattr(self, 'b%s' % key)[layer_idx]
    r = W * x + b

    for i, (child, ri) in enumerate(zip(children, r)):
      U = getattr(self, 'U%s%d' % (key, i))[layer_idx]
      r += U * dy.cmult(child.h, ri)
    return dy.tanh(r)

  def compute_h(self, u, children, i, f):
    r = dy.cmult(u, i)
    for j, (child, fj) in enumerate(zip(children, f)):
      r += dy.cmult(child.h, fj)
    return r

  def compute_output(self, layer_idx, label, children):
    i = self.compute_gate('i', layer_idx, label, children)
    f = [self.compute_gate('f%d' % j, layer_idx, label, children) for j in range(len(children))]
    r = [self.compute_gate('r%d' % j, layer_idx, label, children) for j in range(len(children))]
    u = self.compute_u(layer_idx, label, children, r)
    h = self.compute_h(u, children, i, f)
    return h

  def embed_tree(self, tree: SyntaxTree, layer_idx):
    assert len(tree.children) <= 2
    children = [self.embed_tree(child, layer_idx) for child in tree.children]
    h = self.compute_output(layer_idx, tree.label, children)
    new_tree = SyntaxTree(h, children)
    new_tree.h = h
    return new_tree

class BidirTreeGRU(TreeGRU, Serializable):
  yaml_tag = '!BidirTreeGRU'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               term_encoder=bare(BiLSTMSeqTransducer),
               root_transform=bare(transforms.NonLinear),
               rev_gru=bare(UniGRUSeqTransducer),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    self.layers = layers
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.create_parameters(layers, param_init, bias_init)
    self.root_transform = root_transform
    self.rev_gru = rev_gru

  def embed_tree_inside(self, tree: SyntaxTree, layer_idx):
    assert len(tree.children) <= 2
    children = [self.embed_tree_inside(child, layer_idx) for child in tree.children]

    i = self.compute_gate('i', layer_idx, tree.label, children)
    f = [self.compute_gate('f%d' % j, layer_idx, tree.label, children) for j in range(len(children))]
    r = [self.compute_gate('r%d' % j, layer_idx, tree.label, children) for j in range(len(children))]
    u = self.compute_u(layer_idx, tree.label, children, r)
    h = self.compute_h(u, children, i, f)

    new_tree = SyntaxTree(h, children)
    new_tree.h = h
    return new_tree

  def embed_tree_outside(self, tree: SyntaxTree, layer_idx, parent: dy.Expression):
    if parent == None:
      h = self.root_transform.transform(tree.h)
    else:
      h = self.rev_gru.add_input_to_prev(parent, tree.h)
    children = [self.embed_tree_outside(child, layer_idx, h) for child in tree.children]

    new_tree = SyntaxTree(h, children)
    new_tree.h = h
    return new_tree

  def update_h(self, tree: SyntaxTree):
    tree.h = tree.label
    for child in tree.children:
      self.update_h(child)

  def replace_terms(self, tree: SyntaxTree, terms: expression_seqs.ExpressionSequence):
    if len(tree.children) == 0:
      return SyntaxTree(terms[0], []), terms[1:]

    children = []
    for child in tree.children:
      new_child, terms = self.replace_terms(child, terms)
      children.append(new_child)

    return SyntaxTree(tree.label, children), terms

  def encode_tree(self, tree: SyntaxTree):
    terms = tree.get_terminals()
    terms = expression_seqs.ExpressionSequence(terms)
    encoded_terms = self.term_encoder.transduce(terms)
    tree, r = self.replace_terms(tree, encoded_terms)
    assert len(r) == 0
    return tree

  def embed_tree(self, tree: SyntaxTree, layer_idx):
    encoded = self.encode_tree(tree) if layer_idx == 0 else tree
    inside = self.embed_tree_inside(encoded, layer_idx)
    outside = self.embed_tree_outside(inside, layer_idx, None)
    r = zip_trees([inside, outside])
    self.update_h(r)
    return r

  def shared_params(self):
    return [{".term_encoder.input_dim", ".input_dim"},
            {".term_encoder.hidden_dim", ".input_dim"},
            {".term_encoder.layers", ".layers"}]
