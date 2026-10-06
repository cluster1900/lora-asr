import unittest
from types import SimpleNamespace

from inference.decoding import (
    check_resume_decoding,
    configure_greedy_model,
    decoding_contract,
    positive_token_budget,
    training_token_budget,
)


class DecodingContractTest(unittest.TestCase):
    def test_positive_budget_and_config(self):
        self.assertEqual(positive_token_budget("512"), 512)
        self.assertEqual(training_token_budget({"decoding": {"max_new_tokens": 512}}), 512)
        self.assertEqual(decoding_contract(512)["do_sample"], False)
        with self.assertRaises(ValueError):
            positive_token_budget(0)
        with self.assertRaises(ValueError):
            training_token_budget({"decoding": {"max_new_tokens": 0}})

    def test_configures_official_wrapper_and_generation(self):
        generation = SimpleNamespace()
        model = SimpleNamespace(generation_config=generation, thinker=SimpleNamespace())
        wrapper = SimpleNamespace(model=model, max_new_tokens=128)
        configure_greedy_model(wrapper, 512)
        self.assertEqual(wrapper.max_new_tokens, 512)
        self.assertEqual(generation.max_new_tokens, 512)
        self.assertFalse(generation.do_sample)
        self.assertEqual(generation.num_beams, 1)

    def test_resume_rejects_old_legacy_budget(self):
        check_resume_decoding({"decoding": decoding_contract(512)}, 512)
        with self.assertRaises(ValueError):
            check_resume_decoding({}, 512)


if __name__ == "__main__":
    unittest.main()
