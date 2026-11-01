# Checks that the Gemini API key works and returns schema-valid structured output.
# Skipped when google-genai is not installed or GEMINI_API_KEY is not set (see .env.example),
# so teammates without a key are not blocked. One small request per run.

import os
import unittest

from pydantic import BaseModel

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class BriefCheck(BaseModel):
    client_id: str
    risk_tier: str
    onsite_checks: list[str]


class LlmApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("GEMINI_API_KEY"):
            raise unittest.SkipTest("GEMINI_API_KEY not set")
        try:
            from google import genai
        except ImportError:
            raise unittest.SkipTest("google-genai not installed")
        cls.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def test_structured_output_matches_schema(self):
        response = self.client.models.generate_content(
            model=MODEL,
            contents=(
                "Client train_Client_0 shows an 85% drop in electricity consumption over two "
                "years. Return a short inspection brief for this client."
            ),
            config={
                "response_mime_type": "application/json",
                "response_schema": BriefCheck,
                "temperature": 0,
            },
        )
        brief = BriefCheck.model_validate_json(response.text)
        self.assertEqual(brief.client_id, "train_Client_0")
        self.assertIn(brief.risk_tier.lower(), {"high", "medium", "low"})
        self.assertTrue(brief.onsite_checks)


if __name__ == "__main__":
    unittest.main()
