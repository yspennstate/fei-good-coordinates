# Good coordinates in the Fourier entropy-influence conjecture

Yitzchak Shmalo

This repository contains a paper on the Fourier entropy-influence (FEI) conjecture of Friedgut and Kalai, together
with the scripts behind every numerical value in it.

Paper: [main.pdf](main.pdf) (source [main.tex](main.tex)).

## Summary

For a Boolean function f, let U be drawn from the spectral distribution and put
E_i(f) = H(1[i in U] | U minus i). FEI holds if and only if G = sup_f min_i E_i(f)/Inf_i(f) is finite, and then its
constant lies between G and G + 2/ln 2 ([one-coordinate note](https://github.com/yspennstate/fourier-entropy-influence-one-coordinate)).
The paper studies G directly.

* A composition identity gives E_i and Inf_i for iterated block compositions g(h, ..., h) exactly from the biased
  spectrum of g, carrying four numbers per leaf through each level.
* Towers of transitive monotone gates give G_mon >= 4.50642846544523317790928639..., on about 1.4 x 10^15 variables.
  The values are computed with 40 and 60 digits and certified by 60-digit interval arithmetic.
* Every width-w DNF or CNF has a coordinate with E_i/Inf_i <= 2 + log2 e + log2(2^(w+1) - 1), with no monotonicity
  or read restriction.
* Every relevant leaf of a read-once AND/OR formula has E_i/Inf_i < pi/ln 2, and the constant is sharp.
* The coordinate of largest influence can have unbounded ratio, even for monotone functions; in a monotone
  middle-layer construction all but k of the coordinates have ratio above any constant; and the power-weighted
  averages sum_i Inf_i^a E_i <= C sum_i Inf_i^(a+1) fail for every a != 0 (a = 0 is the conjecture itself).

## Code

The scripts need Python 3, NumPy and mpmath. Saved outputs (`*_out.txt`, `*.txt`) are next to the scripts. In the
outputs, B14 is the 14-gate word in the fifth row of Table 1.

Composition and towers (Sections 2 and 3)

* `code/composition.py` the composition recursion of Theorem 2.1; `code/check_composition.py` checks it against
  brute-force Fourier transforms on 9- to 16-variable compositions.
* `code/towers.py` towers of PM(K33) and its dual.
* `code/record.py` the gates from their DNFs and the tower values in double precision and with 40-digit mpmath;
  `code/table_check.py` every row of Table 1 at 40 and 60 digits, with variable counts.
* `code/convention_scan.py` the sign convention for the gates under which the tower values are reproduced.
* `code/pinned.py` the pinned-mean limit 4.50645546... of the remark after Table 1.
* `code/gates/tower_recompute.py` an independent implementation of the tower recursion;
  `code/gates/tower_iv.py` the same recursion in interval arithmetic with outward rounding (outputs
  `tower_iv_dps60.txt` and `tower_iv_dps60_word2_pd.txt`; the word with dual PM(4) is the one in Table 1);
  `code/gates/realize_gates.py` the search over towers built on the other pinned gates.

Classes and negative results (Sections 4 and 5)

* `code/check_width_bound.py` Lemma 4.1 and Theorem 4.2 on random width-w DNFs with negated literals.
* `code/check_readonce.py`, `code/check_readonce_c.py` Lemma 4.4 against brute force, the integral pi/ln 2, the
  sharpness family and the potential inequalities in the proof of Theorem 4.3.
* `code/check_hidden.py` Proposition 5.1, k = 3, ..., 17.
* `code/check_middle_layer.py` Proposition 5.2, by brute force at k = 3, 5 and by the formula up to k = 17,
  including the base coordinates.
* `code/wa_tribes.py` the power-weight proposition for a > 0: E_x and Inf_x of the tribes substitution against
  full truth tables (k = 3, 19 variables), and the lower bound on the quotient for k = 3, ..., 15;
  `code/wa_tribes2.py` the leaf influence with the normalisation 4p(1-p) (the first block of the output uses the
  unnormalised value) and tribes widths up to 512. Output in `code/wa_tribes_out.txt`.
* `code/check_section5.py` the majority bound (3) in exact fractions, the Laplacian identity and the remark on
  x1 AND (x2 OR x3).

## Related papers

* [The one-coordinate step in the Fourier entropy-influence conjecture](https://github.com/yspennstate/fourier-entropy-influence-one-coordinate)
* [A variance form of Han's bound on the Fourier entropy](https://github.com/yspennstate/fei-variance-han-bound)
* [Four equivalent forms of the Fourier entropy-influence conjecture](https://github.com/yspennstate/fei-equivalent-forms)
* [A filtration bound for the erasure entropies of Boolean functions](https://github.com/yspennstate/fei-filtration-witness)
* [Erasure entropies of sparse linear DNFs](https://github.com/yspennstate/fei-sparse-linear-dnf)
* [The Fourier entropy of linear monotone DNFs on few variables](https://github.com/yspennstate/fei-linear-dnf-census)
* [Hidden coordinates and selection rules in the Fourier entropy-influence conjecture](https://github.com/yspennstate/fei-hidden-coordinates)
* [A forced coordinate with nearly maximal erasure entropy](https://github.com/yspennstate/fei-forced-coordinate)
* [The erasure entropy of a forced coordinate under every bias](https://github.com/yspennstate/fei-forced-coordinate-bias)

## License

The text is released under CC BY 4.0 and the code under the MIT license.
