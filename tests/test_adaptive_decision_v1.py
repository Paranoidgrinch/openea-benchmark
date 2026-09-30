"""No PySCF jobs: only epistemic contracts and boundary tests."""
import unittest
from types import SimpleNamespace

from openea_benchmark.adaptive import (
    DecisionInput, DecisionCategory, PrecisionStatus, DiagnosticID,
    DIAGNOSTIC_CATALOG, EAEstimate, EvidenceQuality, GateSet, Interval,
    Review, ReviewStatus, UncertaintyComponent, ea_from_state_intervals,
    evaluate_decision, evaluate_estimate, ground_state_interval,
    next_diagnostics, stability_diagnostic_from_root,
)


def pass_gate(label):
    return Review(ReviewStatus.CLEARED, evidence_ids=(label + '-record',))


def complete_gates():
    return GateSet(pass_gate('G1'), pass_gate('G2'), pass_gate('G3'))


class TestScientificDecision(unittest.TestCase):
    def test_bound_and_precision_met(self):
        d = evaluate_decision(DecisionInput('OH', Interval(1.20, 1.22), complete_gates()))
        self.assertEqual(d.category, DecisionCategory.BOUND)
        self.assertEqual(d.precision, PrecisionStatus.TARGET_MET)
        self.assertTrue(d.estimated_not_certified)

    def test_bound_sign_but_target_not_met(self):
        d = evaluate_decision(DecisionInput('AlO', Interval(0.20, 0.28), complete_gates()))
        self.assertEqual((d.category, d.precision),
                         (DecisionCategory.BOUND, PrecisionStatus.TARGET_NOT_MET))

    def test_unbound_sign_only_no_need_for_precise_negative_ea(self):
        d = evaluate_decision(DecisionInput('X2', Interval(-2.0, -0.1), complete_gates()))
        self.assertEqual(d.category, DecisionCategory.UNBOUND)
        self.assertEqual(d.precision, PrecisionStatus.NOT_APPLICABLE)

    def test_exactly_zero_upper_bound_is_nonpositive(self):
        d = evaluate_decision(DecisionInput('X2', Interval(-0.5, 0.0), complete_gates()))
        self.assertEqual(d.category, DecisionCategory.UNBOUND)

    def test_ea_near_zero_is_unresolved(self):
        d = evaluate_decision(DecisionInput('X2', Interval(-0.01, 0.01), complete_gates()))
        self.assertEqual(d.category, DecisionCategory.UNRESOLVED)

    def test_no_attachment_evidence_means_no_bound_claim(self):
        gates = GateSet(pass_gate('G1'), Review(ReviewStatus.UNRESOLVED), pass_gate('G3'))
        d = evaluate_decision(DecisionInput('X2', Interval(0.2, 0.3), gates))
        self.assertEqual(d.category, DecisionCategory.UNRESOLVED)
        self.assertIn('G2_ATTACHMENT_RESOLUTION', d.reasons)

    def test_no_state_completeness_means_no_unbound_claim(self):
        gates = GateSet(Review(ReviewStatus.PENDING), pass_gate('G2'), pass_gate('G3'))
        d = evaluate_decision(DecisionInput('X2', Interval(-1.0, -0.3), gates))
        self.assertEqual(d.category, DecisionCategory.UNRESOLVED)

    def test_critical_open_questions_block_a_decision(self):
        d = evaluate_decision(DecisionInput('MgH', Interval(0.1, 0.2), complete_gates(),
                                            critical_open_questions=('UNTESTED_SOC',)))
        self.assertEqual(d.category, DecisionCategory.UNRESOLVED)

    def test_all_empty_gates_never_decide(self):
        self.assertEqual(evaluate_decision(DecisionInput('OH', Interval(1, 2), GateSet())).category,
                         DecisionCategory.UNRESOLVED)

    def test_unknown_correction_is_never_silently_zero(self):
        estimate = EAEstimate(0.1, Interval(-0.01, 0.01), ('base',),
            (UncertaintyComponent('SOC', None, EvidenceQuality.UNKNOWN),))
        d = evaluate_estimate('OH', estimate, complete_gates())
        self.assertEqual(d.category, DecisionCategory.UNRESOLVED)
        self.assertIn('UNKNOWN_CORRECTION:SOC', d.reasons)

    def test_known_correction_is_added_with_interval_arithmetic(self):
        estimate = EAEstimate(0.10, Interval(-0.01, 0.01), ('base',),
            (UncertaintyComponent('ZPE', Interval(0.02, 0.03),
                                  EvidenceQuality.DIRECTLY_TESTED, ('pec1',)),))
        iv = estimate.estimated_interval()
        self.assertAlmostEqual(iv.lower, 0.11)
        self.assertAlmostEqual(iv.upper, 0.14)

    def test_duplicate_correction_names_fail(self):
        a = UncertaintyComponent('SOC', Interval(0, 0.01), EvidenceQuality.DIRECTLY_TESTED, ('s',))
        with self.assertRaises(ValueError):
            EAEstimate(0.3, Interval(0, 0), ('b',), (a, a))

    def test_no_evidence_cannot_be_a_cleared_gate(self):
        with self.assertRaises(ValueError):
            Review(ReviewStatus.CLEARED)

    def test_nonapplicable_requires_rationale(self):
        with self.assertRaises(ValueError):
            Review(ReviewStatus.NOT_APPLICABLE)

    def test_interval_validation(self):
        with self.assertRaises(ValueError):
            Interval(2, 1)
        with self.assertRaises(ValueError):
            Interval(float('nan'), 1)

    def test_ground_state_envelope(self):
        self.assertEqual(ground_state_interval((Interval(1.0, 1.2), Interval(0.9, 1.3))),
                         Interval(0.9, 1.2))

    def test_ea_from_state_ranges(self):
        iv = ea_from_state_intervals((Interval(-2.0, -1.8),), (Interval(-3.0, -2.9),))
        self.assertAlmostEqual(iv.lower, 0.9)
        self.assertAlmostEqual(iv.upper, 1.2)

    def test_all_twelve_diagnostics_declared(self):
        self.assertEqual(len(DIAGNOSTIC_CATALOG), 12)
        self.assertEqual(set(DIAGNOSTIC_CATALOG), set(DiagnosticID))

    def test_scheduler_does_not_rerun_cleared_cases(self):
        reviews = {DiagnosticID.D02_SCF_STABILITY: pass_gate('stable')}
        q = next_diagnostics({DiagnosticID.D02_SCF_STABILITY, DiagnosticID.D08_ATTACHMENT}, reviews)
        self.assertEqual([x.identifier for x in q], [DiagnosticID.D08_ATTACHMENT])

    def test_canonicalized_is_not_automatically_fully_stable(self):
        # Mirrors the existing SCFRootRecord semantics, without PySCF jobs.
        root = SimpleNamespace(root_id='oh-rohf-1', status='CANONICALIZED',
                               internal_stable=True, external_stable=None)
        diagnostic = stability_diagnostic_from_root(root)
        self.assertEqual(diagnostic.review.status, ReviewStatus.UNRESOLVED)

    def test_adapter_can_clear_only_if_both_stabilities_recorded(self):
        root = SimpleNamespace(root_id='x-1', status='CANONICALIZED',
                               internal_stable=True, external_stable=True)
        diagnostic = stability_diagnostic_from_root(root)
        self.assertEqual(diagnostic.review.status, ReviewStatus.CLEARED)

    def test_scheduler_prioritizes_prerequisites(self):
        q = next_diagnostics({DiagnosticID.D11_SOC, DiagnosticID.D04_STATE_COMPETITION}, {})
        self.assertEqual(q[0].identifier, DiagnosticID.D04_STATE_COMPETITION)


if __name__ == '__main__':
    unittest.main()
