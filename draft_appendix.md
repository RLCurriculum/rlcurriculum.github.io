

## Acknowledgments

This template has been adapted from the [World Model template](https://github.com/worldmodels/worldmodels.github.io), written and kindly open-sourced by [David Ha](https://twitter.com/hardmaru).

The experiments in this article were performed on both a P100 GPU and a 64-core CPU Ubuntu Linux virtual machine provided by [Google Cloud Platform](https://cloud.google.com/), using [TensorFlow](https://www.tensorflow.org/).

### Open Source Code

The instructions to reproduce the experiments in this work is available [here](https://github.com/Feryal/jeju_project).

### Reuse

Diagrams and text are licensed under Creative Commons Attribution [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) with the [source available on GitHub](https://github.com/), unless noted otherwise. The figures that have been reused from other sources don’t fall under this license and can be recognized by the citations in their caption.

<h2 id="appendix">Appendix</h2>

In this section we will describe in more details the models and training methods used in this work.


## Bandit Toy Example

As a toy example of how EXP3 work, we can test it on a simple situation with fixed reward allocations:
Consider 3 tasks, providing rewards with fixed probabilities of $p_1=0.2, p_2=0.5, p_3=0.3$. In this situation, the teacher should try these tasks enough time to discover that task 2 is the most valuable.
The evolution of the probability of selecting a task is shown in the figure below.

<div style="text-align: left;">
<img src="assets/exp3_example.png" style="display: block; margin: auto; width: 95%;"/>
<figcaption><b>Example training curve for EXP3 on a toy example.</b><br/>
Given an environment with 3 tasks, with different reward probabilities, EXP3 should collect enough evidence to discover which task to exploit. The plot shows the evolution of the tasks probabilities through time.
</figcaption>
</div>

As one can see, the Teacher explores early, sampling enough time to get a good estimate of the rewards associated with each task.
Then after enough evidence has been collected, it starts exploiting task 2, as one would expect.

<script src="https://gist.github.com/Feryal/ddfe13322e7f2c6186f723f37c444a21.js"></script>

-------------------

## Student Architecture

* Inputs:
  * Observations: Flattened 5x5 egocentric view, 1-hot features & inventory. 1072 features.
  * Task instructions: strings of task names.
* Observation processing:
  2x fully connected with 256 units
* Language processing:
  * Embedding: 20 units
  * LSTM for words: 64 units
* LSTM (recurrent core): 64 units
* Policy:
  Softmax (5 possible actions: Down/Right/Left/Up/Use)
* Value prediction (Critic):
  Linear layer to scalar
-------------------