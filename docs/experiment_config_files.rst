.. _sec-exp-conf:

Experiment configuration file format
====================================

Intro
-----

Configuration files are in `YAML format <https://docs.ansible.com/ansible/YAMLSyntax.html>`_.

At the top-level, a config file consists of a dictionary where keys are experiment
names and values are the experiment specifications. By default, all experiments
are run in lexicographical ordering, but ``xnmt_run_experiments`` can also be told
to run only a selection of the specified experiments. An example template with
2 experiments looks like this

.. code-block:: yaml

    exp1: !Experiment
      exp_global: ...
      preproc: ...
      model: ...
      train: ...
      evaluate: ...
    exp2: !Experiment
      exp_global: ...
      preproc: ...
      model: ...
      train: ...
      evaluate: ...

``!Experiment`` is YAML syntax specifying a Python object of the same name, and
its parameters will be passed on to the Python constructor.
There can be a special top-level entry named ``defaults``; this experiment will
never be run, but can be used as a template where components are partially shared
using YAML anchors or the !Ref mechanism (more on this later).

The usage of ``exp_global``, ``preproc``, ``model``, ``train``, ``evaluate``
are explained below.
Not all of them need to be specified, depending on the use case.


Experiment
~~~~~~~~~~

This specifies settings that are global to this experiment. An example

.. code-block:: yaml

  exp_global: !ExpGlobal
    model_file: '{EXP_DIR}/models/{EXP}.mod'
    log_file: '{EXP_DIR}/logs/{EXP}.log'
    default_layer_dim: 512
    dropout: 0.3

Not that for any strings used here or anywhere in the config file ``{EXP}`` will
be over-written by the name of the experiment, ``{EXP_DIR}`` will be overwritten
by the directory the config file lies in, ``{PID}`` by the process id, and
``{GIT_REV}`` by the current git revision.

To obtain a full list of allowed parameters, please check the documentation for
:ref:`ExpGlobal <mod-experiments>`.

Preprocessing
~~~~~~~~~~~~~

*xnmt* supports a variety of data preprocessing features. Please refer to
:ref:`sec-preproc` for details.

Model
~~~~~
This specifies the model architecture. An typical example looks like this

.. code-block:: yaml

  model: !DefaultTranslator
    src_reader: !PlainTextReader
      vocab: !Vocab {vocab_file: examples/data/head.ja.vocab}
    trg_reader: !PlainTextReader
      vocab: !Vocab {vocab_file: examples/data/head.en.vocab}
    encoder: !BiLSTMSeqTransducer
      layers: 1
    attender: !MlpAttender
      hidden_dim: 512
      state_dim: 512
      input_dim: 512
    decoder: !AutoRegressiveDecoder
      embedder: !SimpleWordEmbedder
        emb_dim: 512
      rnn_layer: !UniLSTMSeqTransducer
        layers: 1
      transform: !NonLinear
        output_dim: 512
      bridge: !CopyBridge {}

The top level entry is typically DefaultTranslator, which implements a standard
attentional sequence-to-sequence model. It allows flexible specification of
encoder, attender, source / target embedder, and other settings. Again, to obtain
the full list of supported options, please refer to the corresponding class
in the :ref:`sec-api-doc`.

Note that some of this Python objects are passed to their parent object's
initializer method, which requires that the children are initialized first.
*xnmt* therefore uses a bottom-up initialization strategy, where siblings
are initialized in the order they appear in the constructor. Among others,
this guarantees that preprocessing is carried out before the model training.

Training
~~~~~~~~

A typical example looks like this

.. code-block:: yaml

  train: !SimpleTrainingRegimen
    trainer: !AdamTrainer
      alpha: 0.001
    run_for_epochs: 2
    src_file: examples/data/head.ja
    trg_file: examples/data/head.en
    dev_tasks:
      - !LossEvalTask
        src_file: examples/data/head.ja
        ref_file: examples/data/head.en

The expected object here is a subclass of TrainingRegimen. Besides
:class:`xnmt.training_regimen.SimpleTrainingRegimen`, multi-task style training regimens are supported.
For multi task training, each training regimen uses their own model, so in this
case models must be specified as sub-components of the training regimen. An example
:ref:`ex-multi-task` configuration can be refered to for more details on this.

Evaluation
~~~~~~~~~~
If specified, the model is tested after training finished.

Config files vs. saved model files
----------------------------------
Saved model files are written out in the exact same YAML format as the config files (with the addition
of some .data directories that contain DyNet weights). This means that it is possible to specify a
saved model as the configuration file. There is one subtle difference: In a config file, placeholders
such as ``{EXP_DIR}`` are resolved based on the current context, which will be different when directly
specifying the saved model file as config file. For this purpose a ``--resume`` option exists that
makes sure to use the context from the saved model file: ``xnmt --resume /path/to/saved-model.mod``.

This feature is currently implemented only in a very basic form: When resuming a crashed experiment,
this will cause the whole experiment to be carried out from the start. When resuming a finished
experiment, *xnmt* will return without performing any action. In the future, this will be extended to
support resuming from the most recent saved checkpoint, etc.

Examples
--------

Here are more elaborate examples from the github repository.

.. _ex-standard:

Standard
~~~~~~~~

.. literalinclude:: examples/01_standard.yaml
    :language: yaml

Minimal
~~~~~~~

.. literalinclude:: examples/02_minimal.yaml
    :language: yaml

Multiple experiments
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: examples/03a_multiple_exp.yaml
    :language: yaml

.. literalinclude:: examples/03b_multiple_exp.yaml
    :language: yaml

.. literalinclude:: examples/03c_multiple_exp.yaml
    :language: yaml

Settings
~~~~~~~~

.. literalinclude:: examples/04_settings.yaml
    :language: yaml

Preprocessing
~~~~~~~~~~~~~

.. literalinclude:: examples/05_preproc.yaml
    :language: yaml

Early stopping
~~~~~~~~~~~~~~

.. literalinclude:: examples/06_early_stopping.yaml
    :language: yaml

Fine-tuning
~~~~~~~~~~~

.. literalinclude:: examples/07_load_finetune.yaml
    :language: yaml

Beam search
~~~~~~~~~~~

.. literalinclude:: examples/08_load_eval_beam.yaml
    :language: yaml

Programmatic usage
~~~~~~~~~~~~~~~~~~

.. literalinclude:: examples/09_programmatic.py
    :language: python

Programmatic loading
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: examples/10_programmatic_load.py
    :language: python

Parameter sharing
~~~~~~~~~~~~~~~~~

.. literalinclude:: examples/11_component_sharing.yaml
    :language: yaml

.. _ex-multi-task:

Multi-task
~~~~~~~~~~

.. literalinclude:: examples/12_multi_task.yaml
    :language: yaml

Speech
~~~~~~

.. literalinclude:: examples/13_speech.yaml
    :language: yaml

Reporting attention matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: examples/14_report.yaml
    :language: yaml

Scoring N-best lists
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: examples/15_score.yaml
    :language: yaml

Transformer
~~~~~~~~~~~

(this is currently broken)

.. literalinclude:: examples/16_transformer.yaml
    :language: yaml

Ensembling
~~~~~~~~~~

.. literalinclude:: examples/17_ensembling.yaml
    :language: yaml

Minimum risk training
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: examples/18_minrisk.yaml
    :language: yaml

Biased Lexicon
~~~~~~~~~~~~~~

(this is currently broken)

.. literalinclude:: examples/19_lexiconbias.yaml
    :language: yaml

Subword Sampling
~~~~~~~~~~~~~~~~

.. literalinclude:: examples/20_subword_sample.yaml
    :language: yaml

Self Attention
~~~~~~~~~~~~~~

.. literalinclude:: examples/21_self_attention.yaml
    :language: yaml

Char Segment
~~~~~~~~~~~~

.. literalinclude:: examples/22_char_segment.yaml
    :language: yaml

Switchout
~~~~~~~~~

.. literalinclude:: examples/23_switchout.yaml
    :language: yaml

Autobatching
~~~~~~~~~~~~

.. literalinclude:: examples/24_autobatch.yaml
    :language: yaml

