# MRF_BCD
This repository contains a Python implementation of the block gradient decent algorithm (BCD) proposed in [Fast MRF Optimization with Application to Depth Reconstruction](http://vladlen.info/papers/fast-mrf.pdf) for Markov Random Field (MRF) optimization.

We give an example by using it to optimize a MRF for binary image segmentation.

We follow [“GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf) to build the workflow. The energy function to be optimized follows [“GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf) and [A Comparative Study of Energy Minimization Methods for Markov Random Fields with Smoothness-Based Priors](https://www.cs.cornell.edu/~rdz/Papers/SZSVKATR-PAMI08.pdf). When the smooth term is not modeled by a Potts model as we do, one may wants also consider implementing the distance transform proposed in this paper for performance improvent (we haven't implemented it here at this moment): [Efficient Belief Propagation for Early Vision](http://cs.brown.edu/people/pfelzens/papers/bp-cvpr.pdf).

Currently the code only works step by step. One may expect further improvement through parellel programming.

To play with it, simply clone the repository to your local machine:
```
git clone https://github.com/zehzhang/MRF_BCD.git
```

Then run:
```
python mrfBCD.py
```

It will print the total energy along with data term and smooth term for each iteration.
When it converges, a plain pixel-wise accuracy by comparing our result with the ground truth will be
printed. An segmentation visualization using the MRF inference will be saved in the name
of `{sample_name}_seg.png` in the same directory. Also, a segmentation visualization based
on purely Bayes inference will be saved in the name of `{sample_name}_bayes.png`. 3 testing
samples are provided.

Below are sample results we obtained:

<div style="color:#0000FF" align="center">
<img src="flower.png" width="430"/> <img src="flower_seg.png" width="430"/>
<img src="person.png" width="430"/> <img src="person_seg.png" width="430"/>
<img src="sponge.png" width="430"/> <img src="sponge_seg.png" width="430"/>
</div>

You may also want to play with some of the parameters. An interesting one would be `lambda`, which balances the data term and the smooth term.

Hope you enjoy it!

If you find any part of this repository useful, consider citing it by:
```
@article{zehuamrfbcd,
    Author = {Zehua Zhang},
    Title = {A Python Implementation of Block Gradient Decent for Optimizing Markov Random Fields},
    Journal = {https://github.com/zehzhang/MRF_BCD},
    Year = {2019}
}
```

If any other materials mentioned here also helps, please do cite them as well! Thanks!
