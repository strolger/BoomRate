# BoomRate

This is the first submission of a volumetric supernova rate code
calulating the volumetric rates and rate histories of different types of
supernovae, and making estimates to yeilds from a hypothetical
survey. Transitioning this previous code to a much more sustainable
version-- Much more work to come!


The main code is rate_calculator.py which takes as an input a .json file
which controls all of the parameters and reference files. The file
"example.json" gives a general idea of the inputs and parameters.

Many of the reference files have hard-coded paths which may have to be
changed to get the code to run completely.

There is a dependecy on SNANA (Kesler+09) supernovae lightcurve and
spectral templates which is not (yet) included here, and a dependency on
filter throughput (transmission) tables, which are also not included
here.

In general, this submission serves only as a reference for how to
calculate SN rates, not yet a fully functioning package. But if you need
some rate calculations, or yeild estimates, feel free to reach out,
strolger@stsci.edu.

-LGS, 20251024


