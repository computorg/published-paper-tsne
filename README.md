# Visualizing Data using t-SNE (mock contribution)
Laurens van der Maaten, Geoffrey Hinton
2008-08-11

*This page is a reworking of the original t-SNE article using the
Computo template. It aims to help authors submitting to the journal by
using some advanced formatting features. We warmly thank the authors of
t-SNE and the editor of JMLR for allowing us to use their work to
illustrate the Computo spirit.*

[![build and
publish](https://github.com/computorg/published-paper-tsne/actions/workflows/build.yml/badge.svg)](https://github.com/computorg/published-paper-tsne/actions/workflows/build.yml)
[![DOI:10.57750/xxxxxx](https://img.shields.io/badge/DOI-10.57750/xxxxxx-034E79.svg)](https://doi.org/10.57750/xxxxxx)
[![Creative Commons
License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

<!-- [![reviews](https://img.shields.io/badge/review-report-blue)](https://github.com/computorg/published-paper-tsne/issues?q=is%3Aopen+is%3Aissue+label%3Areview) -->

### Authors

- [Laurens van der Maaten](https://lvdmaaten.github.io/) (TiCC, Tilburg
  University)
- [Geoffrey Hinton](https://www.cs.toronto.edu/~hinton/) (Department of
  Computer Science, University of Toronto)

### Abstract

We present a new technique called “t-SNE” that visualizes
high-dimensional data by giving each datapoint a location in a two or
three-dimensional map. The technique is a variation of Stochastic
Neighbor Embedding (Hinton and Roweis 2003) that is much easier to
optimize, and produces significantly better visualizations by reducing
the tendency to crowd points together in the center of the map. t-SNE is
better than existing techniques at creating a single map that reveals
structure at many different scales. This is particularly important for
high-dimensional data that lie on several different, but related,
low-dimensional manifolds, such as images of objects from multiple
classes seen from multiple viewpoints. For visualizing the structure of
very large data sets, we show how t-SNE can use random walks on
neighborhood graphs to allow the implicit structure of all the data to
influence the way in which a subset of the data is displayed. We
illustrate the performance of t-SNE on a wide variety of data sets and
compare it with many other non-parametric visualization techniques,
including Sammon mapping, Isomap, and Locally Linear Embedding. The
visualization produced by t-SNE are significantly better than those
produced by other techniques on almost all of the data sets.

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-hinton:stochastic" class="csl-entry">

Hinton, Geoffrey E, and Sam Roweis. 2003. “Stochastic Neighbor
Embedding.” In *Advances in Neural Information Processing Systems*,
edited by S. Becker, S. Thrun, and K. Obermayer. Vol. 15. MIT Press.
<https://proceedings.neurips.cc/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf>.

</div>

</div>
