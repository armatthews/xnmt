"""
This module contains features related to outputs generated by a model.

The main responsibilities are data structures for holding such outputs, and code to translate outputs into readable
strings.
"""

from typing import Optional, Sequence

from xnmt.vocabs import Vocab
from xnmt.persistence import Serializable, serializable_init

class Output(object):
  """
  A template class to represent output generated by a model.
  """
  def __init__(self, actions: Optional[Sequence[int]], score: float) -> None:
    """ Initialize an output with actions.

    Args:
      actions: list of output actions chosen by the model
      score: score given by the model to this output
    """
    self.actions = actions
    self.score = score

  def readable_actions(self) -> Sequence[str]:
    """
    Get a readable version of the output actions.

    This may include looking up word ids in a vocabulary and omitting special tokens such as <s> and </s>

    Returns:
      list containing readable version for each action
    """
    raise NotImplementedError('must be implemented by subclasses')

  def apply_post_processor(self, output_processor: 'OutputProcessor') -> str:
    return output_processor.process_output(self.readable_actions())

  def __str__(self):
    return " ".join(self.readable_actions())

  # for partial compatibility with input objects:
  def sent_len(self):
    return len(self.actions)
  def __iter__(self):
    return iter(self.actions)
  def __getitem__(self, index):
    return self.actions[index]


class ScalarOutput(Output):
  """
  Output of classifier models that generate only a single action.

  Args:
    actions: list of length 1
    score: score given by the model to this output
    vocab: optional vocabulary corresponding to the actions
  """
  def __init__(self, actions: Sequence[int], score: float, vocab: Optional[Vocab] = None) -> None:
    super().__init__(actions=actions, score=score)
    if len(self.actions) > 1: raise ValueError(f"ScalarOutput must have exactly one action, get: {len(self.actions)}")
    self.vocab = vocab

  def readable_actions(self) -> Sequence[str]:
    """
    Get a readable version of the output action by performing on optional vocabulary lookup.

    Returns:
      list containing a single item
    """
    return [self.vocab[self.actions[0]]] if self.vocab else [str(self.actions[0])]


class TextOutput(Output):
  """
  Output of a sequence of actions corresponding to text.

  Args:
    actions: list of length 1
    score: score given by the model to this output
    vocab: optional vocabulary corresponding to the actions
  """
  def __init__(self, actions: Sequence[int], score: float, vocab: Optional[Vocab] = None):
    super().__init__(actions=actions, score=score)
    self.vocab = vocab
    self.filtered_tokens = {Vocab.SS, Vocab.ES}

  def readable_actions(self):
    """
    Get a readable version of the output actions by performing an optional vocabulary lookup and omitting <s> and </s>.

    Returns:
      list containing readable version for each action
    """
    ret = []
    for action in self.actions:
      if action not in self.filtered_tokens:
        ret.append(self.vocab[action] if self.vocab else str(action))
    return ret


class NbestOutput(Output):
  """
  Output in the context of an nbest list.

  Args:
    base_output: The base output object
    nbest_id: The sentence id in the nbest list
    print_score: If True, print nbest_id, score, content separated by ``|||```. If False, drop the score.
  """
  def __init__(self, base_output: Output, nbest_id: int, print_score: bool = False) -> None:
    super().__init__(actions=base_output.actions, score=base_output.score)
    self.base_output = base_output
    self.nbest_id = nbest_id
    self.print_score = print_score
  def readable_actions(self) -> Sequence[str]:
    return self.base_output.readable_actions()
  def __str__(self):
    return self._make_nbest_entry(" ".join(self.readable_actions()))
  def _make_nbest_entry(self, content_str: str) -> str:
    entries = [str(self.nbest_id), content_str]
    if self.print_score:
      entries.insert(1, str(self.base_output.score))
    return " ||| ".join(entries)
  def apply_post_processor(self, output_processor: 'OutputProcessor') -> str:
    return self._make_nbest_entry(output_processor.process_output(self.readable_actions()))

class OutputProcessor(object):
  # TODO: this should be refactored so that multiple processors can be chained
  def process_output(self, output_actions: Sequence) -> str:
    """
    Produce a string-representation of an output.

    Args:
      output_actions: readable output actions

    Returns:
      string representation
    """
    raise NotImplementedError("must be implemented by subclasses")

  @staticmethod
  def get_output_processor(spec):
    if spec == "none":
      return PlainTextOutputProcessor()
    elif spec == "join-char":
      return JoinCharTextOutputProcessor()
    elif spec == "join-bpe":
      return JoinBPETextOutputProcessor()
    elif spec == "join-piece":
      return JoinPieceTextOutputProcessor()
    else:
      return spec

class PlainTextOutputProcessor(OutputProcessor, Serializable):
  """
  Handles the typical case of writing plain text, with one sentence per line.
  """
  yaml_tag = "!PlainTextOutputProcessor"
  def process_output(self, output_actions):
    return " ".join(output_actions)

class JoinCharTextOutputProcessor(PlainTextOutputProcessor, Serializable):
  """
  Assumes a single-character vocabulary and joins them to form words.

  Per default, double underscores '__' are treated as word separating tokens.
  """
  yaml_tag = "!JoinCharTextOutputProcessor"
  @serializable_init
  def __init__(self, space_token="__"):
    self.space_token = space_token

  def process_output(self, output_actions):
    return "".join(" " if s==self.space_token else s for s in  output_actions)

class JoinBPETextOutputProcessor(PlainTextOutputProcessor, Serializable):
  """
  Assumes a bpe-based vocabulary and outputs the merged words.

  Per default, the '@' postfix indicates subwords that should be merged
  """
  yaml_tag = "!JoinBPETextOutputProcessor"
  @serializable_init
  def __init__(self, merge_indicator="@@"):
    self.merge_indicator_with_space = merge_indicator + " "

  def process_output(self, output_actions):
    return " ".join(output_actions).replace(self.merge_indicator_with_space, "")

class JoinPieceTextOutputProcessor(PlainTextOutputProcessor, Serializable):
  """
  Assumes a sentence-piece vocabulary and joins them to form words.

  Space_token could be the starting character of a piece per default, the u'\u2581' indicates spaces
  """
  yaml_tag = "!JoinPieceTextOutputProcessor"
  @serializable_init
  def __init__(self, space_token="\u2581"):
    self.space_token = space_token

  def process_output(self, output_actions):
    return "".join(output_actions).replace(self.space_token, " ").strip()
