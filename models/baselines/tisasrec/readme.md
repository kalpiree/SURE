## TiSASRec

<h2>Overview</h2>

<p>TiSASRec (Time-interval aware Self-Attention Sequential Recommendation) extends SASRec by integrating temporal information into the attention mechanism. It models both item sequences and time intervals between interactions using time-aware embeddings. This implementation supports phase-wise training, allowing continual adaptation across sequential interaction environments.</p>

<h2>Project Structure</h2>

<ul>
<li><code>main.py</code> – Trains and evaluates TiSASRec across five temporal phases for each model instance, storing results and relation matrices.</li>
<li><code>model.py</code> – Defines the TiSASRec model, including time-aware multi-head attention, absolute/relative position embeddings, and pointwise feedforward layers.</li>
<li><code>utils.py</code> – Provides data loading utilities, WarpSampler for negative sampling, relation matrix construction, evaluation functions, and batching.</li>
</ul>

<h2>How to Run</h2>

<h3>Step 1: Train Across Phases</h3>

<pre><code>python main.py \
  --data_dir ./processed_datasets/goodreads/phased_data \
  --output_dir ./output_tisasrec \
  --batch_size 128 \
  --lr 0.001 \
  --device cuda \
  --top_ks 20,20,20,20,20
</code></pre>

<p>This will train TiSASRec across five interaction phases for each user model, saving model checkpoints and evaluation scores in the output directory.</p>


