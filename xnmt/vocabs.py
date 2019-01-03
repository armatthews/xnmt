from typing import Any, List, Optional, Sequence
from collections import namedtuple
import numbers

from xnmt.persistence import serializable_init, Serializable

class Vocab(Serializable):
  """
  An open vocabulary that converts between strings and integer ids.

  The open vocabulary is realized via a special unknown-word token that is used whenever a word is not inside the
  list of known tokens.
  This class is immutable, i.e. its contents are not to change after the vocab has been initialized.

  For initialization, i2w or vocab_file must be specified, but not both.

  Args:
    i2w: complete list of known words, including ``<s>`` and ``</s>``.
    vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    sentencepiece_vocab: Set to ``True`` if ``vocab_file`` is the output of the sentencepiece tokenizer. Defaults to ``False``.
  """

  yaml_tag = "!Vocab"

  SS = 0
  ES = 1

  SS_STR = "<s>"
  ES_STR = "</s>"
  UNK_STR = "<unk>"

  @serializable_init
  def __init__(self,
               i2w: Optional[Sequence[str]] = None,
               vocab_file: Optional[str] = None,
               sentencepiece_vocab: bool = False) -> None:
    assert i2w is None or vocab_file is None
    assert i2w or vocab_file
    if vocab_file:
      i2w = Vocab.i2w_from_vocab_file(vocab_file, sentencepiece_vocab)
    assert i2w is not None
    self.i2w = i2w
    self.w2i = {word: word_id for (word_id, word) in enumerate(self.i2w)}
    if Vocab.UNK_STR not in self.w2i:
      self.w2i[Vocab.UNK_STR] = len(self.i2w)
      self.i2w.append(Vocab.UNK_STR)
    self.unk_token = self.w2i[Vocab.UNK_STR]
    self.save_processed_arg("i2w", self.i2w)
    self.save_processed_arg("vocab_file", None)

  @staticmethod
  def i2w_from_vocab_file(vocab_file: str, sentencepiece_vocab: bool = False) -> List[str]:
    """Load the vocabulary from a file.
    
    If ``sentencepiece_vocab`` is set to True, this will accept a sentencepiece vocabulary file
    
    Args:
      vocab_file: file containing one word per line, and not containing ``<s>``, ``</s>``, ``<unk>``
      sentencepiece_vocab (bool): Set to ``True`` if ``vocab_file`` is the output of the sentencepiece tokenizer. Defaults to ``False``.
    """
    vocab = [Vocab.SS_STR, Vocab.ES_STR]
    reserved = {Vocab.SS_STR, Vocab.ES_STR, Vocab.UNK_STR}
    with open(vocab_file, encoding='utf-8') as f:
      for line in f:
        word = line.strip()
        # Sentencepiece vocab files have second field, ignore it
        if sentencepiece_vocab:
          word = word.split('\t')[0]
        if word in reserved:
          # Ignore if this is a sentencepiece vocab file
          if sentencepiece_vocab:
            continue
          else:
            raise RuntimeError(f"Vocab file {vocab_file} contains a reserved word: {word}")
        vocab.append(word)
    return vocab

  def convert(self, w: str) -> int:
    return self.w2i.get(w, self.unk_token)

  def __getitem__(self, i: numbers.Integral) -> str:
    return self.i2w[i]

  def __len__(self) -> int:
    return len(self.i2w)

  def is_compatible(self, other: Any) -> bool:
    """
    Check if this vocab produces the same conversions as another one.
    """
    if not isinstance(other, Vocab):
      return False
    if len(self) != len(other):
      return False
    if self.unk_token != other.unk_token:
      return False
    return self.w2i == other.w2i

RnngAction = namedtuple('RnngAction', 'action,subaction')

class RnngVocab(Serializable):
  yaml_tag = '!RnngVocab'
  NONE = 0
  SHIFT = 1
  NT = 2
  REDUCE = 3
  NUM_ACTIONS = 4

  @serializable_init
  def __init__(self, term_vocab=None, nt_vocab=None):
    self.term_vocab = term_vocab if term_vocab is not None else Vocab()
    self.nt_vocab = nt_vocab if nt_vocab is not None else Vocab()

  @staticmethod
  def from_vocab_files(term_vocab_file, nt_vocab_file,
                       sentencepiece_vocab=False):
    term_vocab = Vocab(term_vocab_file=term_vocab_file,
                       sentencepiece_vocab=sentencepiece_vocab)
    nt_vocab = Vocab(vocab_file=nt_vocab_file)
    return RnngVocab(term_vocab, nt_vocab)

  def convert(self, word):
    if word == 'REDUCE':
      return RnngAction(RnngVocab.REDUCE, None)
    elif word.startswith('NT(') and word.endswith(')'):
      nt = word.split('(', 1)[1][:-1]
      return RnngAction(RnngVocab.NT, self.nt_vocab.convert(nt))
    elif word.startswith('SHIFT(') and word.endswith(')'):
      term = word.split('(', 1)[1][:-1]
      return RnngAction(RnngVocab.SHIFT, self.term_vocab.convert(term))
    else:
      raise 'Invalid RNNG input word: %s. Should be one of SHIFT(terminal), NT(non-terminal), or REDUCE'

  def __getitem__(self, i):
    assert isinstance(i, RnngAction)
    assert len(i) == 2

    if i[0] == RnngVocab.NONE:
      return 'NONE'
    elif i[0] == RnngVocab.SHIFT:
      return 'SHIFT(%s)' % self.term_vocab[i[1]]
    elif i[0] == RnngVocab.NT:
      return 'NT(%s)' % self.nt_vocab[i[1]]
    elif i[0] == RnngVocab.REDUCE:
      return 'REDUCE'
    raise Exception('Unknown RNNG action: %s' % str(i))

  def __len__(self):
    return len(self.term_vocab) + len(self.nt_vocab)

  def is_compatible(self, other):
    if not isinstance(other, RnngVocab):
      return False
    if not self.term_vocab.is_compatible(other.term_vocab):
      return False
    if not self.nt_vocab.is_compatible(other.nt_vocab):
      return False
