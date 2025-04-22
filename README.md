# PerTDE (Perturbative Tidal Disruption Event)

Code for the two-stage model of Tidal Disruption Events used in the paper <a href="https://arxiv.org/abs/2504.xxxxx">"Modeling Tidal Disruptions with Dynamical Tides"</a> [1].

# Use

Simply run the file <code>TDECalculator_example.ipynb</code> to obtain the density profile of the deformed star, its energy distribution and the mass fallback rate. Choose at the beginning the values of the parameters: the black hole mass, the pericenter distance and the number of steps in the time integration. Change the string <code>MAMS1Msun</code> to change the mass of the star (for example write <code>MAMS0p85Msun</code> for a stellar mass of 0.85 solar masses). The code is simple and easily adjustable to taste. The results for the cases studied in [1] are reported in the folder <code>data-used-in-paper-figures</code>. The folder <code>star-files</code> contains the data on MESA MAMS stars computed with <a href="https://gyre.readthedocs.io/en/stable/">GYRE</a>.

# Questions

For any questions, please send an email to <a href="mailto:zihanz@princeton.edu">zihanz@princeton.edu</a> and <a href="mailto:tomaselli@ias.edu">tomaselli@ias.edu</a>.

# Citing

If you make use of this code, please consider citing the corresponding paper,

<pre><code>@article{Zhou2025:xxx,
    author = "Zhou, Zihan and Tomaselli, Giovanni Maria and Mart\'inez-Rodr\'iguez, Irvin and Li, Jingping",
    title = "{Modeling Tidal Disruptions with Dynamical Tides}",
    eprint = 2504.xxxxx",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    year = "2025"
}
</code></pre>
