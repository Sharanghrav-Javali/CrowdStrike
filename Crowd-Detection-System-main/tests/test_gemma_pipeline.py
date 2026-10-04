import json
import unittest
from types import SimpleNamespace

from gemma_pipeline import (
    InvalidGemmaResponse,
    calculate_grid,
    describe_api_error,
    normalized_box_to_pixels,
    parse_gemma_response,
    result_is_fresh,
    validate_analysis,
)


def valid_payload(**overrides):
    payload = {
        "people_count": 1,
        "persons": [{"box_2d": [100, 200, 900, 500]}],
        "risk_level": "low",
        "crowding_detected": False,
        "crowd_observation": "One person is visible.",
        "recommended_action": "continue_monitoring",
        "reason": "Density is below the threshold.",
    }
    payload.update(overrides)
    return payload


class ParseGemmaResponseTests(unittest.TestCase):
    def test_extracts_function_arguments(self):
        response = SimpleNamespace(function_calls=[SimpleNamespace(
            name="report_crowd_analysis", args=valid_payload()
        )])
        self.assertEqual(parse_gemma_response(response)["people_count"], 1)

    def test_parses_json_text_and_code_fence(self):
        payload = valid_payload()
        response = SimpleNamespace(function_calls=None, text="```json\n" + json.dumps(payload) + "\n```")
        self.assertEqual(parse_gemma_response(response), payload)

    def test_rejects_malformed_json(self):
        response = SimpleNamespace(function_calls=None, text="{broken")
        with self.assertRaises(InvalidGemmaResponse):
            parse_gemma_response(response)

    def test_api_error_has_actionable_status_and_redacts_key(self):
        error = SimpleNamespace(
            code=403,
            status="PERMISSION_DENIED",
            message="Key abc123 was reported as leaked.",
        )
        message = describe_api_error(error, "abc123")
        self.assertIn("403 PERMISSION_DENIED", message)
        self.assertIn("[redacted]", message)
        self.assertNotIn("abc123", message)


class ValidateGemmaAnalysisTests(unittest.TestCase):
    def test_accepts_empty_detection(self):
        result = validate_analysis(valid_payload(people_count=0, persons=[]))
        self.assertEqual(result["people_count"], 0)

    def test_rejects_count_mismatch(self):
        with self.assertRaises(InvalidGemmaResponse):
            validate_analysis(valid_payload(people_count=2))

    def test_rejects_invalid_coordinate_order_or_range(self):
        for box in ([100, 400, 90, 500], [-1, 0, 200, 300], [0, 0, 1001, 300]):
            with self.subTest(box=box), self.assertRaises(InvalidGemmaResponse):
                validate_analysis(valid_payload(persons=[{"box_2d": box}]))

    def test_rejects_non_integer_count(self):
        with self.assertRaises(InvalidGemmaResponse):
            validate_analysis(valid_payload(people_count=True))

    def test_rejects_arbitrary_action(self):
        with self.assertRaises(InvalidGemmaResponse):
            validate_analysis(valid_payload(recommended_action="call_emergency_services"))

    def test_rejects_unknown_risk_level(self):
        with self.assertRaises(InvalidGemmaResponse):
            validate_analysis(valid_payload(risk_level="critical"))


class SpatialAndFreshnessTests(unittest.TestCase):
    def test_converts_yx_normalized_box_to_xy_pixels(self):
        self.assertEqual(normalized_box_to_pixels([100, 200, 900, 500], 1000, 500), (200, 50, 500, 450))

    def test_assigns_person_centers_to_grid_cells(self):
        persons = [
            {"box_2d": [100, 100, 300, 300]},
            {"box_2d": [600, 600, 900, 900]},
        ]
        grid = calculate_grid(persons, 1000, 1000, 4, 4)
        self.assertEqual(grid[0][0], 1)
        self.assertEqual(grid[3][3], 1)
        self.assertEqual(sum(map(sum, grid)), 2)

    def test_rejects_stale_and_future_results(self):
        self.assertTrue(result_is_fresh(98, 100, 3))
        self.assertFalse(result_is_fresh(80, 100, 3))
        self.assertFalse(result_is_fresh(101, 100, 3))


if __name__ == "__main__":
    unittest.main()
