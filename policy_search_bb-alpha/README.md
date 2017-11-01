# PolicySearch-bb-alpha

Python implementation of policy search and model training using  $\alpha$-divergence minimization for Bayesian neural networks with latent variables. See:

Depeweg, Stefan, et al. "Learning and policy search in stochastic dynamical systems with bayesian neural networks." arXiv preprint arXiv:1605.07127 (2016).




### Prerequisites

Requires the standard libraries for theano-based models and Lasagne (I use 0.2.dev)


### Usage


1. Insert industrialbenchmark_python in environment/:

   - [Download](https://github.com/siemens/industrialbenchmark/tree/master/industrial_benchmark_python)  python version of industrialbenchmark.
 
    - Move to environment/industrialbenchmark 


2. Generate batch of state transitions:
 
   ```
    cd  environment/  
    python make_data.py 
    ```

    will generate a training and test set stored in environment/out
    
    X: Setpoint,A(t-14),..,A(t+1),R(t-15)..,R(t) 

    Y: R(t+1)

3. Model Training:

   ```
   cd experiments/  
   python train_model.py 0.5 
    ```

   Will train a BNN using bb-alpha with alpha=0.5 
   
   After training model will be stored in experiments/models as pickle file
   
   Code will run on GPU/CPU. Parameters are chosen conservatively for GPU use. Consider decreasing sample size to 25 for CPU use.
   
   Expected training time (i5-6600K CPU @ 4.0GHz, GTX 1060): 
   CPU:  
   50 samples: 21.5 hours  
   25 samples: 10.5 hours

   GPU:   
   50 samples: 3.5 hours  
    25 samples: 2.0 hours


4. Policy Training

   ```
   cd experiments/  
   python train_controller.py 0.5 
   ```  
   Will train a policy  using model from step 2 (required a model exists in models/)   
   After training the policy will be stored  in experiments/controller as pickle file.
   
   Code will run on CPU. For GPU use one should pass only indexes to train_func using givens. In our experiments no speedup was obtaiend from GPU use.

5. Policy Evaluation:   
   An example policy evaluation script  is given in environment/eval_pol.py   
   ```
   cd environment/  
   ipython  
   from eval_pol import evaluate   
   results = evalute('../experiments/controller/AD_1.0.p')  
   ```

---
Some helpful tips:  

```
model.bb_alpha.network.update_randomness(n_samples)   
```
will  sample  n_samples from q(W) and resample the input noise.

For prediction use:   
```
m,v = model.predict(np.tile(X,[n_samples,1,1]))  
```   
where X is n x d  
 m is n\_samples x n x d   
v is n\_samples x n x d   (constant output noise variance)
