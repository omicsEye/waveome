Artifacts produced under the OLD BIC, which differenced log_posterior_density
(= ELBO + log prior) and so double-counted the prior against BIC's own Occam
term. Superseded by the ELBO-based BIC (commit 416514a, FINDINGS 21-22).

Retained as the evidence behind FINDINGS sections 20-22: the -4.8 / -1.0
log_bf mass points, the 95%-dead SE components, and the demonstration that
the double-count biased the permutation test against live components.

NOT comparable to post-fix outputs. Every log_bf, p-value and q-value here
was computed under the superseded statistic.

ihmp_between_subject.csv is deliberately absent: that test uses Spearman on
subject-level residuals, never touches calc_bic, and remains valid.
