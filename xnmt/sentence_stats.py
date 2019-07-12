from xnmt.persistence import serializable_init, Serializable, bare, Ref

class SourceLengthStat(Serializable):
  yaml_tag = '!SourceLengthStat'

  @serializable_init
  def __init__(self, num_sents=0, trg_len_distribution={}) -> None:
    self.num_sents = num_sents
    self.trg_len_distribution = trg_len_distribution

class TargetLengthStat(Serializable):
  yaml_tag = '!TargetLengthStat'

  @serializable_init
  def __init__(self, num_sents) -> None:
    self.num_sents = num_sents

class SentenceStats(Serializable):
  """
  to Populate the src and trg sents statistics.
  """

  yaml_tag = '!SentenceStats'

  @serializable_init
  def __init__(self, src_stat={}, trg_stat={}, max_pairs=1000000, num_pair=0) -> None:
    self.src_stat = src_stat
    self.trg_stat = trg_stat
    self.max_pairs = max_pairs
    self.num_pair = num_pair

  def add_sent_pair_length(self, src_length, trg_length):
    src_len_stat = self.src_stat.get(src_length, self.SourceLengthStat())
    src_len_stat.num_sents += 1
    src_len_stat.trg_len_distribution[trg_length] = \
      src_len_stat.trg_len_distribution.get(trg_length, 0) + 1
    self.src_stat[src_length] = src_len_stat

    trg_len_stat = self.trg_stat.get(trg_length, self.TargetLengthStat())
    trg_len_stat.num_sents += 1
    self.trg_stat[trg_length] = trg_len_stat

  def populate_statistics(self, train_corpus_src, train_corpus_trg):
    self.num_pair = min(len(train_corpus_src), self.max_pairs)
    for sent_num, (src, trg) in enumerate(zip(train_corpus_src, train_corpus_trg)):
      self.add_sent_pair_length(len(src), len(trg))
      if sent_num > self.max_pairs:
        return
