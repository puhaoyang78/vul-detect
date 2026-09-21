import unittest
from vulnmechanism.source_pilot import class_normalized_weights, check_isolation, project


class PilotTests(unittest.TestCase):
    def test_class_weights_do_not_change_class_total(self):
        rows = [dict(sample_key=str(i), label=i%2) for i in range(8)]
        w = class_normalized_weights(rows, {'0', '2', '1'})
        for label in (0, 1):
            self.assertAlmostEqual(sum(w[r['sample_key']] for r in rows if r['label']==label), 4)
        self.assertAlmostEqual(w['0']/w['4'], 3)
        self.assertAlmostEqual(w['1']/w['3'], 3)

    def test_rejects_initial_fit_group_and_adaptation_project_leakage(self):
        groups = {'train':'a', 'check':'b', 'fit':'c'}
        repos = {'train':'x', 'check':'y', 'fit':'y'}
        check_isolation(['train'], ['check'], ['fit'], groups, repos)
        for fit, g, r in [(['check'], groups, repos),
                           (['fit'], dict(groups, check='c'), repos),
                           (['fit'], groups, dict(repos, check='x'))]:
            with self.assertRaises(ValueError):
                check_isolation(['train'], ['check'], fit, g, r)
        self.assertEqual(project('git.kernel.org/cgit/linux/kernel/git/davem/net'), project('github.com/torvalds/linux'))


if __name__ == '__main__':
    unittest.main()
