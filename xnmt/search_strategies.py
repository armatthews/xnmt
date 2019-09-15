from collections import namedtuple
import math
from typing import Callable, List, Optional, Sequence
import numbers

import dynet as dy
import numpy as np

from xnmt import batchers, logger, vocabs
from xnmt.modelparts import decoders, attenders
from xnmt.length_norm import NoNormalization, LengthNormalization
from xnmt.persistence import Serializable, serializable_init, bare
from xnmt.vocabs import Vocab, RnngVocab
from collections import defaultdict


SearchOutput = namedtuple('SearchOutput', ['word_ids', 'attentions', 'score', 'logsoftmaxes', 'state', 'mask'])
"""
Output of the search
words_ids: list of generated word ids
attentions: list of corresponding attention vector of word_ids
score: a single value of log(p(E|F))
logsoftmaxes: a corresponding softmax vector of the score. score = logsoftmax[word_id]
state: a NON-BACKPROPAGATEABLE state that is used to produce the logsoftmax layer
       state is usually used to generate 'baseline' in reinforce loss
masks: whether the particular word id should be ignored or not (1 for not, 0 for yes)
"""


class SearchStrategy(object):
  """
  A template class to generate translation from the output probability model. (Non-batched operation)
  """
  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      initial_dec_state: decoders.AutoRegressiveDecoderState,
                      initial_att_state: attenders.AttenderState,
                      src_length: Optional[numbers.Integral] = None) -> List[SearchOutput]:
    """
    Args:
      translator: a translator
      initial_dec_state: initial decoder state
      initial_att_state: initial attender state
      src_length: length of src sequence, required for some types of length normalization
    Returns:
      List of (word_ids, attentions, score, logsoftmaxes)
    """
    raise NotImplementedError('generate_output must be implemented in SearchStrategy subclasses')

class GreedySearch(Serializable, SearchStrategy):
  """
  Performs greedy search (aka beam search with beam size 1)

  Args:
    max_len: maximum number of tokens to generate.
  """

  yaml_tag = '!GreedySearch'

  @serializable_init
  def __init__(self, max_len: numbers.Integral = 100) -> None:
    self.max_len = max_len

  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      initial_dec_state: decoders.AutoRegressiveDecoderState,
                      initial_att_state: attenders.AttenderState,
                      src_length: Optional[numbers.Integral] = None) -> List[SearchOutput]:
    # Output variables
    score = []
    word_ids = []
    attentions = []
    logsoftmaxes = []
    states = []
    masks = []
    # Search Variables
    done = None
    current_dec_state = initial_dec_state
    current_att_state = initial_att_state
    for length in range(self.max_len):
      prev_word = word_ids[length-1] if length > 0 else None
      current_output = translator.add_input(prev_word, current_dec_state, current_att_state)
      word_id, word_score = translator.best_k(current_output, 1, normalize_scores=True)
      word_id = word_id[0]
      word_score = word_score[0]
      current_dec_state = current_output.dec_state
      current_att_state = current_output.att_state

      if len(word_id.shape) == 0:
        word_id = np.array([word_id])
        word_score = np.array([word_score])

      if done is not None:
        word_id = [word_id[i] if not done[i] else Vocab.ES for i in range(len(done))]
        mask = [1 if not done[i] else 0 for i in range(len(done))]
        word_score = [s * m for (s, m) in zip(word_score, mask)]
        masks.append(mask)

      # Packing outputs
      score.append(word_score)
      word_ids.append(word_id)
      attentions.append(current_output.attention)
      logsoftmaxes.append(None)
      states.append((current_dec_state, current_att_state))

      # Check if we are done.
      done = [x == Vocab.ES for x in word_id]
      if all(done):
        break

    masks.insert(0, [1 for _ in range(len(done))])
    words = np.stack(word_ids, axis=1)
    score = np.sum(score, axis=0)
    return [SearchOutput(words, attentions, score, logsoftmaxes, states, masks)]

class BeamSearch(Serializable, SearchStrategy):
  """
  Performs beam search.

  Args:
    beam_size: number of beams
    max_len: maximum number of tokens to generate.
    len_norm: type of length normalization to apply
    one_best: Whether to output the best hyp only or all completed hyps.
    scores_proc: apply an optional operation on all scores prior to choosing the top k.
                 E.g. use with :class:`xnmt.length_normalization.EosBooster`.
  """

  yaml_tag = '!BeamSearch'
  Hypothesis = namedtuple('Hypothesis', ['score', 'output', 'parent', 'word'])

  @serializable_init
  def __init__(self,
               beam_size: numbers.Integral = 1,
               max_len: numbers.Integral = 100,
               len_norm: LengthNormalization = bare(NoNormalization),
               one_best: bool = True,
               scores_proc: Optional[Callable[[np.ndarray], None]] = None) -> None:
    self.beam_size = beam_size
    self.max_len = max_len
    self.len_norm = len_norm
    self.one_best = one_best
    self.scores_proc = scores_proc

  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      initial_dec_state: decoders.DecoderState,
                      initial_att_state: attenders.AttenderState,
                      src_length: Optional[numbers.Integral] = None) -> List[SearchOutput]:

    # TODO(philip30): can only do single decoding, not batched
    active_hyp = [self.Hypothesis(0, None, None, None)]
    completed_hyp = []
    for length in range(self.max_len):
      if len(completed_hyp) >= self.beam_size:
        completed_hyp = sorted(completed_hyp, key=lambda hyp: hyp.score, reverse=True)
        completed_hyp = completed_hyp[:self.beam_size]
        worst_complete_hyp_score = completed_hyp[-1].score
        active_hyp = [hyp for hyp in active_hyp if hyp.score >= worst_complete_hyp_score]
        # Assumption: each additional word will always *decrease* the total score.
        if len(active_hyp) == 0:
          break

      # Expand hyp
      new_set = []
      for hyp in active_hyp:
        # Note: prev_word has *not* yet been added to prev_state
        if length > 0:
          prev_word = hyp.word
          prev_dec_state = hyp.output.dec_state
          prev_att_state = hyp.output.att_state
        else:
          prev_word = None
          prev_dec_state = initial_dec_state
          prev_att_state = initial_att_state

        current_output = translator.add_input(prev_word, prev_dec_state, prev_att_state)
        # We have a complete hypothesis
        if current_output.dec_state.is_complete():
          completed_hyp.append(hyp)
          continue

        # Find the k best words at the next time step
        top_words, top_scores = translator.best_k(current_output, self.beam_size, normalize_scores=True)
        assert len(top_words) == len(top_scores)
        assert len(top_words) > 0

        # Queue next states 
        for cur_word, score in zip(top_words, top_scores):
          assert len(score.shape) == 0
          new_score = self.len_norm.normalize_partial_topk(hyp.score, score, length + 1)
          new_set.append(self.Hypothesis(new_score, current_output, hyp, cur_word))

      # Next top hypothesis
      active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.beam_size]

    # There is no hyp that reached </s>
    if len(completed_hyp) == 0:
      assert len(active_hyp) > 0
      completed_hyp = active_hyp

    # Length Normalization
    normalized_scores = self.len_norm.normalize_completed(completed_hyp, src_length)
    hyp_and_score = sorted(list(zip(completed_hyp, normalized_scores)), key=lambda x: x[1], reverse=True)

    # Take only the one best, if that's what was desired
    if self.one_best:
      hyp_and_score = hyp_and_score[:1]

    return self.backtrace(hyp_and_score)

  def backtrace(self, hyp_and_score):
    # Backtracing + Packing outputs
    results = []
    for end_hyp, score in hyp_and_score:
      logsoftmaxes = []
      word_ids = []
      attentions = []
      states = []
      current = end_hyp
      while current.parent is not None:
        word_ids.append(current.word)
        attentions.append(current.output.attention)
        states.append((current.output.dec_state, current.output.att_state))
        current = current.parent
      results.append(SearchOutput([list(reversed(word_ids))], [list(reversed(attentions))],
                                  [score], list(reversed(logsoftmaxes)),
                                  list(reversed(states)), [1 for _ in word_ids]))
    return results

class RnngBagBeamSearch(Serializable, SearchStrategy):
  """
  Performs "bag-level" beam search for RNNG output.
  See "Neural Generative Rhetorical Structure Parsing" (Mabona et al., 2019).
  """

  yaml_tag = '!RnngBagBeamSearch'
  # Translator output contains dec state, att state, and attention
  Hypothesis = namedtuple('Hypothesis', ['score', 'output', 'parent', 'word'])

  @serializable_init
  def __init__(self,
               beam_size: numbers.Integral = 1,
               max_len: numbers.Integral = 100,
               len_norm: LengthNormalization = bare(NoNormalization),
               one_best: bool = True,
               scores_proc: Optional[Callable[[np.ndarray], None]] = None) -> None:
    self.beam_size = beam_size
    self.max_len = max_len
    self.len_norm = len_norm
    self.one_best = one_best
    self.scores_proc = scores_proc

  def is_prunable(self, score, bucket, presorted=False):
    """ Returns True iff a hypothesis with score ``score'' has a chance to make
    it into the top k of the hypothesis bin ``bucket''"""
    if len(bucket) < self.beam_size:
      return False

    sorted_bucket = bucket if presorted else sorted(bucket, key=lambda hyp: hyp.score, reverse=True)
    return score < sorted_bucket[self.beam_size - 1].score

  def prune(self, hyps, bucket):
    bucket = sorted(bucket, key=lambda hyp: hyp.score, reverse=True)
    hyps = [hyp for hyp in hyps if not self.is_prunable(hyp.score, bucket, presorted=True)]
    return hyps

  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      initial_dec_state: decoders.DecoderState,
                      initial_att_state: attenders.AttenderState,
                      src_length: Optional[numbers.Integral] = None) -> List[SearchOutput]:
    hyp_bins = defaultdict(lambda: defaultdict(list))
    hyp_bins[0][0].append(self.Hypothesis(0, None, None, None))
    completed_hyp = []

    # Observation: No prefix can ever have more shifts than NTs (unless it's a complete hyp)
    #for num_term_actions in range((self.max_len - 1) // 2):
    #  for num_struct_actions in range((self.max_len + 1) // 2):
    for num_struct_actions in range((self.max_len + 1) // 2):
      for num_term_actions in range(num_struct_actions + 2):
        print('Bin (%d, %d): %d hyps' % (num_term_actions, num_struct_actions, len(hyp_bins[num_term_actions][num_struct_actions])))
        # This heuristic dropped us from 25.8479 BLEU down to 24.23 BLEU
        if self.one_best and len(completed_hyp) > 0 and False:
          best_complete_score = max([hyp.score for hyp in completed_hyp])
          hyp_bins[num_term_actions][num_struct_actions] = [hyp for hyp in hyp_bins[num_term_actions][num_struct_actions] if hyp.score >= best_complete_score]
        elif len(completed_hyp) >= self.beam_size:
          # This branch drops us from 25.8479 to 25.7082.
          # This combined with the one below got 25.6885 BLEU.
          print('Completed scores: ' + str([hyp.score for hyp in completed_hyp]))
          print('This bin: ' + str([hyp.score for hyp in hyp_bins[num_term_actions][num_struct_actions]]))
          hyp_bins[num_term_actions][num_struct_actions] = self.prune(hyp_bins[num_term_actions][num_struct_actions], completed_hyp)
          print('Pruned down to %d hyps' % (len(hyp_bins[num_term_actions][num_struct_actions])))

        hyp_bins[num_term_actions][num_struct_actions] = sorted(hyp_bins[num_term_actions][num_struct_actions], key=lambda x: x.score, reverse=True)[:self.beam_size]
        for hyp in hyp_bins[num_term_actions][num_struct_actions]:

          # On German--English s2t these pruning criteria hurt BLEU only very slightly, from 25.8479 down to 25.8328.
          if self.is_prunable(hyp.score, hyp_bins[num_term_actions + 1][num_struct_actions]) and self.is_prunable(hyp.score, hyp_bins[num_term_actions][num_struct_actions + 1]):
            continue

          if num_term_actions > 0 or num_struct_actions > 0:
            prev_word = hyp.word
            prev_dec_state = hyp.output.dec_state
            prev_att_state = hyp.output.att_state
          else:
            prev_word = None
            prev_dec_state = initial_dec_state
            prev_att_state = initial_att_state

          current_output = translator.add_input(prev_word, prev_dec_state, prev_att_state)
          if current_output.dec_state.is_complete():
            assert num_term_actions == num_struct_actions + 1
            print('Complete hyp!')
            completed_hyp.append(hyp)
            continue
          else:
            assert num_term_actions < num_struct_actions + 1

          print('Expanding hyp ' + str(hyp.word) + '\t' + str(hyp.score))
          top_words, top_scores = translator.best_k(current_output, self.beam_size, normalize_scores=True)
          for cur_word, score in zip(top_words, top_scores):
            new_score = self.len_norm.normalize_partial_topk(hyp.score, score, num_term_actions + num_struct_actions + 1)
            new_hyp = self.Hypothesis(new_score, current_output, hyp, cur_word)
            if cur_word.action == vocabs.RnngVocab.SHIFT:
              print('  -> ' + str(cur_word) + '\t' + str(score) + ' to bin (%d, %d)' % (num_term_actions + 1, num_struct_actions))
              hyp_bins[num_term_actions + 1][num_struct_actions].append(new_hyp)
            else:
              print('  -> ' + str(cur_word) + '\t' + str(score) + ' to bin (%d, %d)' % (num_term_actions, num_struct_actions + 1))
              hyp_bins[num_term_actions][num_struct_actions + 1].append(new_hyp)

    # There is no hyp that reached </s>
    if len(completed_hyp) == 0:
      assert False

    # Length Normalization
    normalized_scores = self.len_norm.normalize_completed(completed_hyp, src_length)
    hyp_and_score = sorted(list(zip(completed_hyp, normalized_scores)), key=lambda x: x[1], reverse=True)

    # Take only the one best, if that's what was desired
    if self.one_best:
      hyp_and_score = hyp_and_score[:1]

    # Backtracing + Packing outputs
    results = []
    for end_hyp, score in hyp_and_score:
      logsoftmaxes = []
      word_ids = []
      attentions = []
      states = []
      current = end_hyp
      while current.parent is not None:
        word_ids.append(current.word)
        attentions.append(current.output.attention)
        states.append((current.output.dec_state, current.output.att_state))
        current = current.parent
      results.append(SearchOutput([list(reversed(word_ids))], [list(reversed(attentions))],
                                  [score], list(reversed(logsoftmaxes)),
                                  list(reversed(states)), [1 for _ in word_ids]))
    return results

class RnngBeamSearch(Serializable, SearchStrategy):
  """
  Performs beam search for RNNG output. See https://arxiv.org/pdf/1707.08976.pdf.
  """

  yaml_tag = '!RnngBeamSearch'
  # Translator output contains dec state, att state, and attention
  Hypothesis = namedtuple('Hypothesis', ['score', 'output', 'parent', 'word'])

  @serializable_init
  def __init__(self,
               beam_size: numbers.Integral = 1,
               max_len: numbers.Integral = 100,
               len_norm: LengthNormalization = bare(NoNormalization),
               one_best: bool = True,
               scores_proc: Optional[Callable[[np.ndarray], None]] = None) -> None:
    self.beam_size = beam_size
    self.max_len = max_len
    self.len_norm = len_norm
    self.one_best = one_best
    self.scores_proc = scores_proc

  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      initial_dec_state: decoders.DecoderState,
                      initial_att_state: attenders.AttenderState,
                      src_length: Optional[numbers.Integral] = None) -> List[SearchOutput]:
    # Normal beam search sorts hypotheses just by the current number of actions taken.
    # This sorts hypotheses first by the number of _terminals_ produced so far, and then
    # secondarily by the number of structure actions taken since the last shift.
    # The search starts by looking at the (0, 0) bucket.
    # At step (i, j), we take the top k next action for each hypothesis in the (i, j) bin.
    # If the next action is a shift, the resulting hypotheis is added to the (i+1, 0) bin.
    # Otherwise, the resultuing hypothesis is added to the (i, j+1) bin.

    initial_output = translator.add_input(None, initial_dec_state, initial_att_state)
    active_hyp = [self.Hypothesis(0, None, None, None)]
    completed_hyp = []

    for length in range(self.max_len):
      if len(completed_hyp) >= self.beam_size:
        completed_hyp = sorted(completed_hyp, key=lambda hyp: hyp.score, reverse=True)
        completed_hyp = completed_hyp[:self.beam_size]
        worst_complete_hyp_score = completed_hyp[-1].score
        active_hyp = [hyp for hyp in active_hyp if hyp.score >= worst_complete_hyp_score]
        # Assumption: each additional word will always *decrease* the total score.
        if len(active_hyp) == 0:
          break

      # Expand hyp
      #print('Clearing new shift set')
      new_shift_set = []
      for num_struct_actions in range(0, self.max_len - length + 1):
        #print('Doing bucket (%d, %d) with %d active hyps' % (length, num_struct_actions, len(active_hyp)))
        #for hyp in active_hyp:
        #  print('  - ' + str(hyp.score) + '\t' + str(hyp.word))
        fasttrack = (len(new_shift_set) == 0)
        new_struct_set = []
        for hyp in active_hyp:
          if length > 0 or num_struct_actions > 0:
            prev_word = hyp.word
            prev_dec_state = hyp.output.dec_state
            prev_att_state = hyp.output.att_state
          else:
            prev_word = None
            prev_dec_state = initial_dec_state
            prev_att_state = initial_att_state

          current_output = translator.add_input(prev_word, prev_dec_state, prev_att_state)
 
          if current_output.dec_state.is_complete():
            #print('Completed hyp!')
            completed_hyp.append(hyp)
            continue

          # Here we "fasttrack" the top few terminals directly into the (i+1, 0) bucket.
          # This avoids the catastrophe that happens when the model may have only NTs in its k-best lists for many timesteps in a row
          if fasttrack:
            top_words, top_scores = translator.decoder.best_k(current_output.dec_state, self.beam_size, normalize_scores=True, shifts_only=True)
            #print('Fast tracking %d words' % len(top_words))
            for cur_word, score in zip(top_words, top_scores):
              assert len(score.shape) == 0
              new_score = self.len_norm.normalize_partial_topk(hyp.score, score, length + 1)
              new_hyp = self.Hypothesis(new_score, current_output, hyp, cur_word)
              #print('Fast tracking ' + str(score) + '\t' + str(cur_word))
              new_shift_set.append(new_hyp)

          top_words, top_scores = translator.best_k(current_output, self.beam_size, normalize_scores=True)
          assert len(top_words) == len(top_scores)
          assert len(top_words) > 0

          for cur_word, score in zip(top_words, top_scores):
            assert len(score.shape) == 0
            if cur_word.action == vocabs.RnngVocab.SHIFT:
              new_score = self.len_norm.normalize_partial_topk(hyp.score, score, length + 1)
            else:
              new_score = hyp.score + score
            new_hyp = self.Hypothesis(new_score, current_output, hyp, cur_word)

            #print(str(score) + '\t' + str(cur_word))
            if cur_word.action == vocabs.RnngVocab.SHIFT:
              if not fasttrack:
                new_shift_set.append(new_hyp)
              pass
            else:
              new_struct_set.append(new_hyp)

        active_hyp = sorted(new_struct_set, key=lambda x: x.score, reverse=True)[:self.beam_size]

        if len(new_shift_set) >= self.beam_size:
          new_shift_set = sorted(new_shift_set, key=lambda x: x.score, reverse=True)[:self.beam_size]
          assert len(new_shift_set) == self.beam_size
          for i in range(len(active_hyp)):
            if active_hyp[i].score < new_shift_set[-1].score:
              #print('Active hyp #%d has score %f. Cannot catch up to %dth best shift action, which has a score of %f. %sing...' % (i, active_hyp[i].score, len(new_shift_set), new_shift_set[-1].score, 'Break' if i == 0 else 'Prun'))
              active_hyp = active_hyp[:i]
              break
        if len(active_hyp) == 0:
          break 

      if len(new_shift_set) == 0:
        assert len(completed_hyp) > 0
      else:
        assert len(new_shift_set) >= self.beam_size, 'Shift set only has %d candidates.' % len(new_shift_set)
        active_hyp = sorted(new_shift_set, key=lambda x: x.score, reverse=True)[:self.beam_size]

    # There is no hyp that reached </s>
    if len(completed_hyp) == 0:
      assert len(active_hyp) > 0
      completed_hyp = active_hyp

    # Length Normalization
    normalized_scores = self.len_norm.normalize_completed(completed_hyp, src_length)
    hyp_and_score = sorted(list(zip(completed_hyp, normalized_scores)), key=lambda x: x[1], reverse=True)

    # Take only the one best, if that's what was desired
    if self.one_best:
      hyp_and_score = hyp_and_score[:1]

    # Backtracing + Packing outputs
    results = []
    for end_hyp, score in hyp_and_score:
      logsoftmaxes = []
      word_ids = []
      attentions = []
      states = []
      current = end_hyp
      while current.parent is not None:
        word_ids.append(current.word)
        attentions.append(current.output.attention)
        states.append((current.output.dec_state, current.output.att_state))
        # TODO(philip30): This should probably be uncommented.
        # These 2 statements are an overhead because it is need only for reinforce and minrisk
        # Furthermore, the attentions is only needed for report.
        # We should have a global flag to indicate whether this is needed or not?
        # The global flag is modified if certain objects is instantiated.
        #logsoftmaxes.append(dy.pick(current.output.logsoftmax, current.word))
        #states.append(translator.get_nobp_state(current.output.state))
        current = current.parent
      results.append(SearchOutput([list(reversed(word_ids))], [list(reversed(attentions))],
                                  [score], list(reversed(logsoftmaxes)),
                                  list(reversed(states)), [1 for _ in word_ids]))
    return results

class SamplingSearch(Serializable, SearchStrategy):
  """
  Performs search based on the softmax probability distribution.
  Similar to greedy searchol

  Args:
    max_len:
    sample_size:
  """

  yaml_tag = '!SamplingSearch'

  @serializable_init
  def __init__(self, max_len: numbers.Integral = 100, sample_size: numbers.Integral = 5) -> None:
    self.max_len = max_len
    self.sample_size = sample_size

  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      initial_dec_state: decoders.AutoRegressiveDecoderState,
                      initial_att_state: attenders.AttenderState,
                      src_length: Optional[numbers.Integral] = None) -> List[SearchOutput]:
    outputs = []
    for k in range(self.sample_size):
      outputs.append(self.sample_one(translator, initial_dec_state, initial_att_state))
    return outputs

  # Words ids, attentions, score, logsoftmax, state
  def sample_one(self,
                 translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                 initial_dec_state: decoders.AutoRegressiveDecoderState,
                 initial_att_state: attenders.AttenderState) -> SearchOutput:
    # Search variables
    current_words = None
    current_dec_state = initial_dec_state
    current_att_state = initial_att_state
    done = None
    # Outputs
    scores = []
    samples = []
    states = []
    attentions = []
    masks = []
    # Sample to the max length
    for length in range(self.max_len):
      current_output = translator.add_input(current_words, current_dec_state, current_att_state)
      if current_output.dec_state.is_complete():
        break
      word_id, word_score = translator.sample(current_output, 1)[0]
      word_score = word_score.npvalue()
      assert word_score.shape == (1,)
      word_score = word_score[0]

      if type(word_id) != np.array or len(word_id.shape) == 0:
        word_id = batchers.mark_as_batch([word_id])
        word_score = np.array([word_score])

      if done is not None:
        word_id = [word_id[i] if not done[i] else Vocab.ES for i in range(len(done))]
        # masking for logsoftmax
        mask = [1 if not done[i] else 0 for i in range(len(done))]
        word_score = [s * m for (s, m) in zip(word_score, mask)]
        masks.append(mask)

      # Appending output
      scores.append(word_score)
      samples.append(word_id)
      states.append(current_output)
      attentions.append(current_output.attention)

      # Next time step
      current_words = word_id
      current_dec_state = current_output.dec_state
      current_att_state = current_output.att_state

    # Packing output
    scores = [np.sum(scores)]
    masks.insert(0, [1 for _ in range(len(word_id))])
    samples = np.stack(samples, axis=1)
    return SearchOutput(samples, attentions, scores, [None for _ in samples], states, masks)


class MctsNode(object):
  def __init__(self,
               parent: Optional['MctsNode'],
               prior_dist: np.ndarray,
               word: Optional[numbers.Integral],
               attention: Optional[List[np.ndarray]],
               translator: 'xnmt.models.translators.AutoRegressiveTranslator',
               dec_state: decoders.AutoRegressiveDecoderState) -> None:
    self.parent = parent
    self.prior_dist = prior_dist  # log of softmax
    self.word = word
    self.attention = attention

    self.translator = translator
    self.dec_state = dec_state

    self.tries = 0
    self.avg_value = 0.0
    self.children = {}

    # If the child is unvisited, set its avg_value to
    # parent value - reduction where reduction = c * sqrt(sum of scores of all visited children)
    # where c is 0.25 in leela
    self.reduction = 0.0

  def choose_child(self) -> numbers.Integral:
    return max(range(len(self.prior_dist)), key=lambda move: self.compute_priority(move))

  def compute_priority(self, move: numbers.Integral) -> numbers.Real:
    if move not in self.children:
      child_val = self.prior_dist[move] + self.avg_value - self.reduction
      child_tries = 0
    else:
      child_val = self.prior_dist[move] + self.children[move].avg_value
      child_tries = self.children[move].tries

    K = 5.0
    exp_term = math.sqrt(1.0 * self.tries + 1.0) / (child_tries + 1)
    # TODO: This exp could be done before the prior is passed into the MctsNode
    # so it's done as a big batch
    exp_term *= K * math.exp(self.prior_dist[move])
    total_value = child_val + exp_term
    return total_value

  def expand(self) -> 'MctsNode':
    if self.word == Vocab.ES:
      return self

    move = self.choose_child()
    if move in self.children:
      return self.children[move].expand()
    else:
      output = self.translator.add_input(move, self.dec_state)
      prior_dist = self.translator.calc_log_probs(output.state).npvalue()
      attention = output.attention

      path = []
      node = self
      while node is not None:
        path.append(node.word)
        node = node.parent
      path = ' '.join(str(word) for word in reversed(path))
      print('Creating new node:', path, '+', move)
      new_node = MctsNode(self, prior_dist, move, attention,
                          self.translator, output.state)
      self.children[move] = new_node
      return new_node

  def rollout(self, sample_func, max_len):
    prefix = []
    scores = []
    prev_word = None
    dec_state = self.dec_state

    if self.word == Vocab.ES:
      return prefix, scores

    while True:
      output = self.translator.add_input(prev_word, dec_state)
      logsoftmax = self.translator.calc_log_probs(output.state).npvalue()
      attention = output.attention
      best_id = sample_func(logsoftmax)
      print("Rolling out node with word=", best_id, 'score=', logsoftmax[best_id])

      prefix.append(best_id)
      scores.append(logsoftmax[best_id])

      if best_id == Vocab.ES or len(prefix) >= max_len:
        break
      prev_word = best_id
      dec_state = output.state
    return prefix, scores

  def backup(self, result):
    print('Backing up', result)
    self.avg_value = self.avg_value * (self.tries / (self.tries + 1)) + result / (self.tries + 1)
    self.tries += 1
    if self.parent is not None:
      my_prob = self.parent.prior_dist[self.word]
      self.parent.backup(result + my_prob)

  def collect(self, words, attentions):
    if self.word is not None:
      words.append(self.word)
      attentions.append(self.attention)
    if len(self.children) > 0:
      best_child = max(self.children.itervalues(), key=lambda child: child.visits)
      best_child.collect(words, attentions)


def random_choice(logsoftmax: np.ndarray) -> numbers.Integral:
  #logsoftmax *= 100
  probs = np.exp(logsoftmax)
  probs /= sum(probs)
  choices = np.random.choice(len(probs), 1, p=probs)
  return choices[0]


def greedy_choice(logsoftmax: np.ndarray) -> numbers.Integral:
  return np.argmax(logsoftmax)


class MctsSearch(Serializable, SearchStrategy):
  """
  Performs search with Monte Carlo Tree Search
  """
  yaml_tag = '!MctsSearch'

  @serializable_init
  def __init__(self, visits: numbers.Integral = 200, max_len: numbers. Integral = 100) -> None:
    self.max_len = max_len
    self.visits = visits

  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      dec_state: decoders.AutoRegressiveDecoderState,
                      att_state: attenders.AttenderState,
                      src_length: Optional[numbers.Integral] = None) -> List[SearchOutput]:
    orig_dec_state = dec_state

    output = translator.add_input(None, dec_state)
    dec_state = output.state
    assert dec_state == orig_dec_state
    logsoftmax = self.translator.calc_log_probs(dec_state).npvalue()
    root_node = MctsNode(None, logsoftmax, None, None, translator, dec_state)
    for i in range(self.visits):
      terminal = root_node.expand()
      words, scores = terminal.rollout(random_choice, self.max_len)
      terminal.backup(sum(scores))
      print()

    print('Final stats:')
    for word in root_node.children:
      print (word, root_node.compute_priority(word), root_node.prior_dist[word] + root_node.children[word].avg_value, root_node.children[word].tries)
    print()

    scores = []
    logsoftmaxes = []
    word_ids = []
    attentions = []
    states = []
    masks = []

    node = root_node
    while True:
      if len(node.children) == 0:
        break
      best_word = max(node.children, key=lambda word: node.children[word].tries)
      score = node.prior_dist[best_word]
      attention = node.children[best_word].attention

      scores.append(score)
      logsoftmaxes.append(node.prior_dist)
      word_ids.append(best_word)
      attentions.append(attention)
      states.append(node.dec_state)
      masks.append(1)

      node = node.children[best_word]

    word_ids = np.expand_dims(word_ids, axis=0)
    return [SearchOutput(word_ids, attentions, scores, logsoftmaxes, states, masks)]
