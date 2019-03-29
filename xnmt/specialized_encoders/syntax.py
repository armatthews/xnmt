import sys
import numbers
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import dynet as dy

from xnmt import expression_seqs, param_collections, param_initializers, batchers
from xnmt.modelparts import transforms, attenders
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.transducers import base as transducers
from xnmt.transducers import recurrent
from xnmt.persistence import bare, Ref, Serializable, serializable_init, Path
from xnmt.sent import SyntaxTree


def linearize(tree: SyntaxTree):
  """Converts a SyntaxTree of vectors into a linear sequence of vectors"""
  r = [tree.label]
  for child in tree.children:
    r += linearize(child)
  return expression_seqs.ExpressionSequence(r)

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
      recurrent.UniLSTMSeqTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init[i] if isinstance(param_init, Sequence) else param_init,
                           bias_init=bias_init[i] if isinstance(bias_init, Sequence) else bias_init) for i in
      range(layers)])

    self.inside_rev_layers = self.add_serializable_component("inside_rev_layers", inside_rev_layers, lambda: [
      recurrent.UniLSTMSeqTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init[i] if isinstance(param_init, Sequence) else param_init,
                           bias_init=bias_init[i] if isinstance(bias_init, Sequence) else bias_init) for i in
      range(layers)])

    self.outside_left_layers = self.add_serializable_component("outside_left_layers", outside_left_layers, lambda: [
      recurrent.UniLSTMSeqTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init[i] if isinstance(param_init, Sequence) else param_init,
                           bias_init=bias_init[i] if isinstance(bias_init, Sequence) else bias_init) for i in
      range(layers)])

    self.outside_right_layers = self.add_serializable_component("outside_right_layers", outside_right_layers, lambda: [
      recurrent.UniLSTMSeqTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout,
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
    from xnmt import batchers
    self.batch_size = src.batch_size() if batchers.is_batched(src) else 1

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    # TODO: Real final states
    z = dy.zeros(self.hidden_dim, batch_size=self.batch_size)
    return [transducers.FinalTransducerState(z, z)]

  def compute_gate(self, key, layer_idx, x, children, activation=dy.logistic):
    W = getattr(self, 'W%s' % key)[layer_idx]
    b = getattr(self, 'b%s' % key)[layer_idx]
    r = W * x + b

    for i, child in enumerate(children):
      U = getattr(self, 'U%s%d' % (key, i))[layer_idx]
      r += U * child.h

    return activation(r)

  def transduce(self, trees: Union[SyntaxTree, List[SyntaxTree]]) -> expression_seqs.ExpressionSequence:
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
               term_encoder=bare(recurrent.BiLSTMSeqTransducer),
               root_transform=bare(transforms.NonLinear),
               rev_gru=bare(recurrent.UniGRUSeqTransducer),
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
      assert len(h) == 1
      h = h[0]
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

class BatchedBidirTreeGRU(TreeGRU, Serializable):
  yaml_tag = '!BatchedBidirTreeGRU'

  # TODO: root_transform and rev_gru should be lists of stuff, one per layer
  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               term_encoder=bare(recurrent.BiLSTMSeqTransducer),
               root_transform=bare(transforms.NonLinear),
               rev_gru=bare(recurrent.UniGRUSeqTransducer),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    self.layers = layers
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.create_parameters(layers, param_init, bias_init)
    self.root_transform = root_transform
    self.rev_gru = rev_gru

  def transduce(self, trees: batchers.SyntaxTreeBatch) -> expression_seqs.ExpressionSequence:
    for layer_idx in range(self.layers):
      trees = self.embed_tree(trees, layer_idx)
    return linearize(trees)

  def transform_single(self, tree: SyntaxTree, term_vecs, nt_vecs, term_idx, nt_idx, out_nodes, out_vectors):
    tree.idx = term_idx + nt_idx
    assert len(out_nodes) == tree.idx
    assert len(out_vectors) == tree.idx
    out_nodes.append(tree)
    if len(tree.children) == 0:
      assert term_idx < term_vecs.dim()[1]
      out_vectors.append(dy.pick_batch_elem(term_vecs, term_idx))
      term_idx += 1
    else:
      assert nt_idx < nt_vecs.dim()[1]
      out_vectors.append(dy.pick_batch_elem(nt_vecs, nt_idx))
      nt_idx += 1
      for child in tree.children:
        term_idx, nt_idx = self.transform_single(child, term_vecs, nt_vecs, term_idx, nt_idx, out_nodes, out_vectors)

    assert len(out_nodes) == len(out_vectors)
    return term_idx, nt_idx

  def transform(self, batch: batchers.SyntaxTreeBatch):
    nodes = []
    vectors = []
    for tree in batch.trees:
      n = []
      v = []
      self.transform_single(tree, batch.leaves, batch.non_leaves, 0, 0, n, v)
      nodes.append(n)
      vectors.append(v)
    return nodes, vectors

  def dtl_sort_single(self, tree: SyntaxTree, term_vecs, nt_vecs, stuff, term_idx, nt_idx, node_vectors):
    tree.idx = term_idx + nt_idx
    assert len(node_vectors) == tree.idx
    if len(tree.children) == 0:
      assert term_idx < term_vecs.dim()[1]
      node_vectors.append(dy.pick_batch_elem(term_vecs, term_idx))
      term_idx += 1
      tree.dtl = 0
      arity = 0
    else:
      assert nt_idx < nt_vecs.dim()[1]
      node_vectors.append(dy.pick_batch_elem(nt_vecs, nt_idx))
      nt_idx += 1
      for child in tree.children:
        term_idx, nt_idx = self.dtl_sort_single(child, term_vecs, nt_vecs, stuff, term_idx, nt_idx, node_vectors)
      tree.dtl = max([child.dtl for child in tree.children]) + 1
      arity = len(tree.children)
    print('At level %d nodes %s build into node %s' % (tree.dtl, str([child.idx for child in tree.children]), str(tree.idx)))
    stuff[tree.dtl][arity].append((tree.idx, [child.idx for child in tree.children]))
    return term_idx, nt_idx

  def dtr_sort_single(self, tree: SyntaxTree, term_vecs, nt_vecs, stuff, term_idx, nt_idx, node_vectors, dtr=0, parent_idx=-1):
    tree.idx = term_idx + nt_idx
    assert len(node_vectors) == tree.idx
    if len(tree.children) == 0:
      assert term_idx < term_vecs.dim()[1]
      node_vectors.append(dy.pick_batch_elem(term_vecs, term_idx))
      term_idx += 1
      tree.dtr = dtr
      arity = 0
    else:
      assert nt_idx < nt_vecs.dim()[1]
      node_vectors.append(dy.pick_batch_elem(nt_vecs, nt_idx))
      nt_idx += 1
      tree.dtr = dtr
      arity = len(tree.children)
      for child in tree.children:
        term_idx, nt_idx = self.dtr_sort_single(child, term_vecs, nt_vecs, stuff, term_idx, nt_idx, node_vectors, dtr + 1, tree.idx)
    print('Node %d dtr is %d, parent is %d' % (tree.idx, dtr, parent_idx))
    stuff[dtr].append((tree.idx, parent_idx))
    return term_idx, nt_idx

  def dtl_sort(self, batch: batchers.SyntaxTreeBatch, stuff=defaultdict(lambda: defaultdict(list)), term_idx=0, nt_idx=0, node_vectors=[]):
    print('Term vecs: ' + str(batch.leaves.dim()))
    print('NT vecs: ' + str(batch.non_leaves.dim()))
    for tree in batch.trees:
      term_idx, nt_idx = self.dtl_sort_single(tree, batch.leaves, batch.non_leaves, stuff, term_idx, nt_idx, node_vectors)
    return stuff, node_vectors

  def dtr_sort(self, batch: batchers.SyntaxTreeBatch, stuff=defaultdict(list), term_idx=0, nt_idx=0, node_vectors=[]):
    for tree in batch.trees:
      term_idx, nt_idx = self.dtr_sort_single(tree, batch.leaves, batch.non_leaves, stuff, term_idx, nt_idx, node_vectors)
    return stuff, node_vectors

  def compute_gate(self, key, layer_idx, x, children, activation=dy.logistic):
    W = getattr(self, 'W%s' % key)[layer_idx]
    b = getattr(self, 'b%s' % key)[layer_idx]
    r = W * x + b

    for i, child in enumerate(children):
      U = getattr(self, 'U%s%d' % (key, i))[layer_idx]
      r += U * child

    return activation(r)

  def compute_u(self, layer_idx, x, children, r):
    key = 'u'
    W = getattr(self, 'W%s' % key)[layer_idx]
    b = getattr(self, 'b%s' % key)[layer_idx]
    r = W * x + b

    for i, (child, ri) in enumerate(zip(children, r)):
      U = getattr(self, 'U%s%d' % (key, i))[layer_idx]
      r += U * dy.cmult(child, ri)
    return dy.tanh(r)

  def compute_h(self, u, children, i, f):
    r = dy.cmult(u, i)
    for j, (child, fj) in enumerate(zip(children, f)):
      r += dy.cmult(child, fj)
    return r

  def untransform_single(self, tree, node_vectors, leaves, non_leaves, term_idx, nt_idx):
    idx = term_idx + nt_idx
    print('Untransforming node %d (%d/%d). Array lengths are %d, %d, and %d' % (idx, term_idx, nt_idx, len(node_vectors), len(leaves), len(non_leaves)))
    assert idx < len(node_vectors)
    assert term_idx == len(leaves)
    assert nt_idx == len(non_leaves)
    if len(tree.children) == 0:
      leaves.append(node_vectors[term_idx])
      term_idx += 1
    else:
      non_leaves.append(node_vectors[nt_idx])
      nt_idx += 1
      for child in tree.children:
        term_idx, nt_idx = self.untransform_single(child, node_vectors, leaves, non_leaves, term_idx, nt_idx)
    return term_idx, nt_idx

  def untransform(self, trees, node_vectors, term_idx=0, nt_idx=0):
    leaves = []
    non_leaves = []
    for tree in trees:
      term_idx, nt_idx = self.untransform_single(tree, node_vectors, leaves, non_leaves, term_idx, nt_idx)
    assert term_idx + nt_idx == len(node_vectors)
    return dy.concatenate_to_batch(leaves), dy.concatenate_to_batch(non_leaves)

  def embed_tree_inside(self, batch: batchers.SyntaxTreeBatch, layer_idx):
    # Topologically sort the trees in the batch
    stuff, node_vectors = self.dtl_sort(batch)

    # For each level (0 = terminals), and arity
    # create a lists of nodes that need computed.
    for level in sorted(stuff.keys()):
      for arity in sorted(stuff[level].keys()):
        labels = []
        children = [[] for _ in range(arity)]
        for p, cs in stuff[level][arity]:
          assert len(cs) == arity
          labels.append(node_vectors[p])
          for i, child_idx in enumerate(cs):
            children[i].append(node_vectors[child_idx])

        # Turn the lists of stuff into actual batches
        labels = dy.concatenate_to_batch(labels)
        children = [dy.concatenate_to_batch(children_i) for children_i in children]

        # Compute the TreeGRU stuff
        i = self.compute_gate('i', layer_idx, labels, children)
        f = [self.compute_gate('f%d' % j, layer_idx, labels, children) for j in range(len(children))]
        r = [self.compute_gate('r%d' % j, layer_idx, labels, children) for j in range(len(children))]
        u = self.compute_u(layer_idx, labels, children, r)
        h = self.compute_h(u, children, i, f)
        assert len(stuff[level][arity]) == h.dim()[1]

        # Update the node_vectors array with the results
        for i, (p, _) in enumerate(stuff[level][arity]):
          v = dy.pick_batch_elem(h, i)
          node_vectors[p] = v

    leaves, non_leaves = self.untransform(batch.trees, node_vectors)
    return batchers.SyntaxTreeBatch(batch.trees, leaves, non_leaves)

  def embed_tree_outside(self, batch: batchers.SyntaxTreeBatch, layer_idx):
    # Topologically sort the trees in the batch
    stuff, node_vectors = self.dtr_sort(batch)

    # For each level (0 = root), and arity
    # create a lists of nodes that need computed.
    for level in sorted(stuff.keys()):
      labels = []
      parents = []
      for idx, parent_idx in stuff[level]:
        assert (level == 0) == (parent_idx == -1)
        labels.append(node_vectors[idx])
        if level != 0:
          parents.append(node_vectors[parent_idx])

      # Turn the lists of stuff into actual batches
      labels = dy.concatenate_to_batch(labels)
      if level != 0:
        parents = dy.concatenate_to_batch(parents)

      # Run the outside GRU to get new embeddings
      print('DTR %d:' % level)
      if level != 0:
        print('Parents: ' + str(parents.dim()))
      print('Labels: ' + str(labels.dim()))
      if level != 0:
        h = self.rev_gru.add_input_to_prev(parents, labels)
        assert len(h) == 1
        h = h[0]
      else:
        h = self.root_transform.transform(labels)
      print('h: %s' % (str(type(h))))

      # Update the node_vectors array with the results
      for i, (p, _) in enumerate(stuff[level]):
        v = dy.pick_batch_elem(h, i)
        node_vectors[p] = v

    leaves, non_leaves = self.untransform(batch.trees, node_vectors)
    return batchers.SyntaxTreeBatch(batch.trees, leaves, non_leaves)

  def encode_tree(self, batch: batchers.SyntaxTreeBatch):
    term_seqs = []
    term_index = 0
    print('Sent lengths: ' + str([len(tree.get_terminals()) for tree in batch.trees]))
    print('Old leaves:' + str(batch.leaves.dim()))
    for tree in batch.trees:
      # First a bit of bounds checking
      num_terms = len(tree.get_terminals())
      assert term_index + num_terms <= batch.leaves.dim()[1]

      # Create a terminal sequence for this tree
      # starting with <s> and ending with </s>
      term_seqs.append([])
      term_seqs[-1].append(batch.SS)
      for i in range(term_index, term_index + num_terms):
        term_seqs[-1].append(dy.pick_batch_elem(batch.leaves, i))
      term_seqs[-1].append(batch.ES)

      # Update start index
      term_index += num_terms

    # Done with loop. Make sure everything went as expected.
    assert term_index == batch.leaves.dim()[1]

    # Pad the terminal sequences to the max length
    longest = max([len(seq) for seq in term_seqs]) 
    for i in range(len(term_seqs)):
      padding = longest - len(term_seqs[i])
      term_seqs[i] = term_seqs[i] + [batch.ES] * padding
    batched = []
    for i in range(longest):
      yay = dy.concatenate_to_batch([seq[i] for seq in term_seqs])
      batched.append(yay)
    batched = expression_seqs.ExpressionSequence(batched)
    print('batched: ' + str(batched.dim()))

    # Transduce the terminals using the encoder (usually a BiLSTM)
    encoded_terms = self.term_encoder.transduce(batched)
    new_leaves = []
    for tree in batch.trees:
      num_terms = len(tree.get_terminals())
      x = encoded_terms[1:num_terms + 1]
      x = [dy.pick_batch_elem(xj, i) for xj in x]
      new_leaves += x 
    new_leaves = dy.concatenate_to_batch(new_leaves)
    print('New leaves: ' + str(new_leaves.dim()))

    r = batchers.SyntaxTreeBatch(batch.trees, new_leaves, batch.non_leaves)
    return r

  def combine_trees(self, batches: List[batchers.SyntaxTreeBatch]) -> batchers.SyntaxTreeBatch:
    for batch in batches[1:]:
      assert batch.trees == batches[0].trees

    leaves = dy.esum([batch.leaves for batch in batches])
    non_leaves = dy.esum([batch.non_leaves for batch in batches])
    return batchers.SyntaxTreeBatch(batches[0].trees, leaves, non_leaves)

  def embed_tree(self, batch: batchers.SyntaxTreeBatch, layer_idx):
    encoded = self.encode_tree(batch) if layer_idx == 0 else tree
    inside = self.embed_tree_inside(encoded, layer_idx)
    outside = self.embed_tree_outside(inside, layer_idx)
    combined = self.combine_trees([inside, outside])
    return combined

  def linearize(self, batch: batchers.SyntaxTreeBatch) -> expression_seqs.ExpressionSequence:
    nodes, vectors = self.transform(batch)
    assert len(nodes) == len(vectors)

    print('v[0][0]: ' + str(vectors[0][0].dim()))
    zeros = dy.zeros(vectors[0][0].dim()[0])
    print('zeros: ' + str(zeros.dim()))
    max_len = max([len(v) for v in vectors])

    for i in range(len(nodes)):
      n = nodes[i]
      v = vectors[i]
      assert len(n) == len(v)
      v += [zeros] * (max_len - len(v))

    for v in vectors:
      assert len(v) == max_len

    print('Vectors is ' + str(len(vectors)) + ' by ' + str([len(v) for v in vectors]))
    for v in vectors:
      for x in v:
        print(x.dim())

    steps = [[vectors[i][j] for i in range(len(vectors))] for j in range(max_len)]
    steps = [dy.concatenate_to_batch(s) for s in steps]
    steps = expression_seqs.ExpressionSequence(steps)
    print('Steps: ' + str(steps.dim()))
    return steps

  def transduce(self, batch: batchers.SyntaxTreeBatch) -> expression_seqs.ExpressionSequence:
    for layer_idx in range(self.layers):
      batch = self.embed_tree(batch, layer_idx)
    return self.linearize(batch)

  def shared_params(self):
    return [{".term_encoder.input_dim", ".input_dim"},
            {".term_encoder.hidden_dim", ".input_dim"},
            {".term_encoder.layers", ".layers"}]

