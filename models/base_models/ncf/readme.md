
## NCF

<h2>Overview</h2>

<p>NCF (Neural Collaborative Filtering) is a deep learning framework combining Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) to model user–item interactions. This implementation supports training via phased data and incremental learning across phases.</p>

<h2>Project Structure</h2>

<ul>
<li><code>model.py</code> - Defines the NCF model supporting GMF, MLP, NeuMF-end, and NeuMF-pre variants.</li>
<li><code>data_utils.py</code> - Utilities to load data and generate training datasets with negative sampling.</li>
<li><code>evaluate.py</code> - Contains standard evaluation functions (HR and NDCG).</li>
<li><code>evaluate_csv.py</code> - Performs evaluation using candidate lists from CSV (supports partial item sets).</li>
<li><code>train_ncf.py</code> - Main training script for phased learning across multiple datasets and phases.</li>
<li><code>main.py</code> - (Legacy) One-shot training script used for original NCF experiments.</li>
<li><code>config.py</code> - Dataset/model configuration paths.</li>
<li><code>utils.py</code> - Seed setting and miscellaneous utilities.</li>
</ul>

<h2>How to Run</h2>

<h3>Step 1: Train Across Phases</h3>

<p>Use the following command to train NCF on phased data:</p>

<pre><code>python train_ncf.py
</code></pre>

<p>This script will iterate over all datasets and models under <code>processed_datasets/*</code> and train through multiple phases (e.g., phase0 to phase4), optionally resuming from previous checkpoints.</p>



