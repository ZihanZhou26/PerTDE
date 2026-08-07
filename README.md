# PerTDE (Perturbative Tidal Disruption Event)

Code for the updated two-stage model of Tidal Disruption Events with relativistic corrections used in the paper <!-- # <a href="https://arxiv.org/abs/2504.16025"> --> "Modeling Relativistic Tidal Disruptions of MESA Stars"</a> [1].


# Use

Simply run the file <code>TDECalculator_example.ipynb</code> to obtain the density profile of the deformed star. To obtain its energy distribution and the mass fallback rate, pre-computed distributions for some example stars are provided and plotted in <code>FallbackGen_example.ipynb</code>. More pre-calculated distributions are saved in the folder <code>Fallback_Data</code>

Choose at the beginning the values of the parameters: a Newtonian or relativistic orbit, the black hole mass, the pericenter distance, the black hole spin, and the number of steps in the time integration. Change the string <code>MAMS1Msun</code> to change the mass of the star (for example write <code>MAMS0p85Msun</code> for a stellar mass of 0.85 solar masses). The code is simple and easily adjustable to taste. To save energy distribution and mass fallback rate calculations, <code>prograde_dT.ipynb</code>, <code>retrograde_dT.ipynb</code>, <code>sch_dT.ipynb</code> are provided to easily generate prograde, retrograde, Schwarzschild, and Newtonian energy distribution and fallback rate calculations, which are automatically stored in <code>Fallback_Data</code>. 

The results for the cases studied in [1] are reported in the folder <code>data-used-in-paper-figures</code>. The folder <code>star-files</code> contains the data on MESA MAMS stars computed with <a href="https://gyre.readthedocs.io/en/stable/">GYRE</a>.

# Questions

For any questions, please send an email to <a href="mailto:lwang959@wisc.edu">lwang959@wisc.edu</a>, <a href="mailto:zihanz@princeton.edu">zihanz@princeton.edu</a>, and <a href="mailto:tomaselli@ias.edu">tomaselli@ias.edu</a>.

# Citing

If you make use of this code, please consider citing the corresponding paper,

<!-- <pre><code>@article{Zhou:2025lzg,
    author = "Zhou, Zihan and Tomaselli, Giovanni Maria and Mart\'\i{}nez-Rodr\'\i{}guez, Irvin and Li, Jingping",
    title = "{Modeling Tidal Disruptions with Dynamical Tides}",
    eprint = "2504.16025",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "4",
    year = "2025"
}
</code></pre> -->
