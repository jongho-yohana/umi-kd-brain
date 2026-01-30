#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalBrainConfig(BaseModel):
    evaluator_model: str  # e.g., "azure/gpt-4o" or "anthropic/claude-sonnet-4-20250514"
    responder_model: str  # e.g., "anthropic/claude-sonnet-4-20250514" or "azure/gpt-4o"
    evaluator_temperature: float = 0.1  # Lower = more deterministic analysis
    responder_temperature: float = 0.3  # Higher = more creative responses
    max_tokens: int = 4000  # Maximum tokens for each model call


class ClinicalBrainMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str  # Message text content


class ClinicalBrainResponse(BaseModel):
    user_name: str = "Umi"
    message: str = ""  # Main wellness coaching response
    metadata: Dict[str, Any] = {}  # Optional: response_category, confidence, etc.


class ClinicalBrainRunner:

    def __init__(self, config: ClinicalBrainConfig):
        self.config = config

        prompt_dir = Path(__file__).parent / "src/umi/agent/runtime/agents/clinical_brain"
        self.evaluation_prompt = self._load_prompt(prompt_dir / "prompt_evaluation.txt")
        self.responder_prompt = self._load_prompt(prompt_dir / "prompt_responder.txt")

        logger.info(f"Initialized ClinicalBrain: evaluator={config.evaluator_model}, responder={config.responder_model}")

    def _load_prompt(self, prompt_path: Path) -> str:
        """Load prompt from file"""
        try:
            with prompt_path.open("r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"Loaded prompt from {prompt_path}")
                return content
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise

    async def _call_llm(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3
    ) -> str:
        """Call LLM using litellm"""
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM {model}: {e}")
            raise

    def _convert_messages(self, messages: List[ClinicalBrainMessage]) -> List[Dict[str, str]]:
        """Convert ClinicalBrainMessage to dict format for LLM"""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    async def evaluate_messages(self, messages: List[ClinicalBrainMessage]) -> str:
        logger.info(f"Running evaluation with {self.config.evaluator_model}")

        evaluation_messages = [
            {"role": "system", "content": self.evaluation_prompt}
        ]
        evaluation_messages.extend(self._convert_messages(messages))

        start_time = time.time()
        evaluation_result = await self._call_llm(
            model=self.config.evaluator_model,
            messages=evaluation_messages,
            temperature=self.config.evaluator_temperature
        )
        elapsed_time = time.time() - start_time

        logger.info(f"Evaluation completed in {elapsed_time:.2f}s")
        return evaluation_result

    async def generate_response(
        self,
        messages: List[ClinicalBrainMessage],
        evaluation_result: str
    ) -> str:
        logger.info(f"Generating response with {self.config.responder_model}")

        responder_prompt = self.responder_prompt.replace(
            "{structured_analysis}",
            evaluation_result
        )

        response_messages = [
            {"role": "system", "content": responder_prompt}
        ]
        response_messages.extend(self._convert_messages(messages))

        start_time = time.time()
        response = await self._call_llm(
            model=self.config.responder_model,
            messages=response_messages,
            temperature=self.config.responder_temperature
        )
        elapsed_time = time.time() - start_time

        logger.info(f"Response generated in {elapsed_time:.2f}s")
        return response

    async def process_conversation(
        self,
        messages: List[ClinicalBrainMessage]
    ) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("Processing conversation through clinical brain")

        total_start = time.time()

        # Step 1: Evaluate messages (Stage 1)
        eval_start = time.time()
        evaluation_result = await self.evaluate_messages(messages)
        eval_time = time.time() - eval_start

        logger.info("EVALUATION OUTPUT:")
        logger.info("-" * 80)
        logger.info(evaluation_result)
        logger.info("-" * 80)

        # Step 2: Generate response (Stage 2)
        resp_start = time.time()
        full_response = await self.generate_response(messages, evaluation_result)
        resp_time = time.time() - resp_start

        logger.info("RESPONSE OUTPUT:")
        logger.info("-" * 80)
        logger.info(full_response)
        logger.info("-" * 80)

        total_time = time.time() - total_start

        # Parse response
        try:
            response_data = json.loads(full_response)
            parsed_response = ClinicalBrainResponse(**response_data)
        except json.JSONDecodeError:
            logger.warning("Response is not valid JSON, returning as plain text")
            parsed_response = ClinicalBrainResponse(message=full_response)

        return {
            "evaluation_result": evaluation_result,
            "response": parsed_response.dict(),
            "metrics": {
                "evaluation_time_s": eval_time,
                "response_time_s": resp_time,
                "total_time_s": total_time
            },
            "config": {
                "evaluator_model": self.config.evaluator_model,
                "responder_model": self.config.responder_model
            }
        }


class DataCollector:

    def __init__(self, output_dir: str = "clinical_brain_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Configure litellm
        self._setup_litellm()

        # Define model combinations
        # Based on umi-cai container.py configuration
        # Production: GPT-4o (eval) + Claude-4 (resp) performs best
        self.model_configs = [
            ClinicalBrainConfig(
                evaluator_model="azure/gpt-4o",
                responder_model="anthropic/claude-sonnet-4-20250514"
            ),
            ClinicalBrainConfig(
                evaluator_model="anthropic/claude-sonnet-4-20250514",
                responder_model="azure/gpt-4o"
            ),
        ]

        logger.info(f"Data collector initialized with {len(self.model_configs)} model combinations")
        logger.info(f"Output directory: {self.output_dir}")

    def _setup_litellm(self):
        """Setup LiteLLM with environment variables"""
        # Set Azure environment variables
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_api_base = os.getenv("AZURE_API_BASE")
        azure_api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")

        if azure_api_key:
            os.environ["AZURE_API_KEY"] = azure_api_key
        if azure_api_base:
            os.environ["AZURE_API_BASE"] = azure_api_base
        if azure_api_version:
            os.environ["AZURE_API_VERSION"] = azure_api_version

        # Set Anthropic environment variables
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

        # Configure LiteLLM settings
        litellm.drop_params = True

        logger.info("LiteLLM configured successfully")

    def get_test_cases(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "wellness_snacks",
                "description": "Simple wellness question about healthy snacks",
                "category": "nutrition",  # For CSV post-processing
                "subcategory": "healthy_snacks",
                "messages": [
                    ClinicalBrainMessage(
                        role="user",
                        content="Hi Umi, what are some healthy snacks for kids?"
                    )
                ]
            },
            {
                "id": "family_goal_eating_healthy",
                "description": "Family goal setting for healthy eating with time constraints",
                "category": "nutrition",
                "subcategory": "meal_planning",
                "messages": [
                    ClinicalBrainMessage(
                        role="user",
                        content="Our family wants to start eating healthier but we're always so busy with work and school. Do you have any suggestions that don't take too much time to prepare?"
                    )
                ]
            },
            {
                "id": "stress_and_family",
                "description": "Multi-message conversation about work stress affecting family",
                "category": "stress",
                "subcategory": "work_life_balance",
                "messages": [
                    ClinicalBrainMessage(
                        role="user",
                        content="I've been feeling really overwhelmed with work lately"
                    ),
                    ClinicalBrainMessage(
                        role="user",
                        content="It's affecting how I interact with my family. What should I do?"
                    )
                ]
            },
            {
                "id": "child_anxiety_sleep",
                "description": "Child behavior question about anxiety and sleep issues",
                "category": "sleep",
                "subcategory": "child_sleep_anxiety",
                "messages": [
                    ClinicalBrainMessage(
                        role="user",
                        content="My 8-year-old has been having trouble sleeping and seems anxious about school. Any suggestions?"
                    )
                ]
            },
            {
                "id": "family_activity_planning",
                "description": "Family activity planning with specific age constraints",
                "category": "physical_activity",
                "subcategory": "family_activities",
                "messages": [
                    ClinicalBrainMessage(
                        role="user",
                        content="We want to be more active as a family but don't know where to start. We have kids ages 6 and 12."
                    )
                ]
            },
            {
                "id": "meal_planning_picky_eaters",
                "description": "Meal planning with picky eaters challenge",
                "category": "nutrition",
                "subcategory": "picky_eaters",
                "messages": [
                    ClinicalBrainMessage(
                        role="user",
                        content="I'm trying to get my family to eat more vegetables but my kids are really picky eaters. Any tips?"
                    )
                ]
            }
        ]

    async def collect_data(self, test_case_ids: Optional[List[str]] = None):
        """Collect data from test cases using all model combinations"""
        test_cases = self.get_test_cases()

        if test_case_ids:
            test_cases = [tc for tc in test_cases if tc["id"] in test_case_ids]

        logger.info(f"Starting data collection for {len(test_cases)} test cases")
        logger.info(f"Using {len(self.model_configs)} model combinations")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = []

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Test Case {i}/{len(test_cases)}: {test_case['id']}")
            logger.info(f"Description: {test_case['description']}")
            logger.info(f"{'=' * 80}\n")

            test_case_results = {
                "test_case": test_case,
                "timestamp": timestamp,
                "results": []
            }

            for j, config in enumerate(self.model_configs, 1):
                logger.info(f"\nModel Combination {j}/{len(self.model_configs)}:")
                logger.info(f"  Evaluator: {config.evaluator_model}")
                logger.info(f"  Responder: {config.responder_model}")

                try:
                    runner = ClinicalBrainRunner(config)
                    result = await runner.process_conversation(test_case["messages"])

                    test_case_results["results"].append({
                        "success": True,
                        "data": result
                    })

                    logger.info(f"✅ Successfully processed with this model combination")

                except Exception as e:
                    logger.error(f"❌ Error processing with this model combination: {e}")
                    test_case_results["results"].append({
                        "success": False,
                        "error": str(e),
                        "config": config.dict()
                    })

                # Add delay between API calls to avoid rate limiting
                await asyncio.sleep(2)

            all_results.append(test_case_results)

            # Save individual test case results
            output_file = self.output_dir / f"{test_case['id']}_{timestamp}.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(test_case_results, f, indent=2, ensure_ascii=False)
            logger.info(f"\n💾 Results saved to: {output_file}")

        # Save summary results
        summary_file = self.output_dir / f"summary_{timestamp}.json"
        summary = {
            "timestamp": timestamp,
            "total_test_cases": len(test_cases),
            "model_combinations": [config.dict() for config in self.model_configs],
            "results": all_results
        }

        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ Data collection complete!")
        logger.info(f"💾 Summary saved to: {summary_file}")
        logger.info(f"{'=' * 80}\n")

        self._print_summary(all_results)

    def _print_summary(self, all_results: List[Dict[str, Any]]):
        """Print summary of collected data"""
        total_tests = len(all_results)
        total_runs = sum(len(r["results"]) for r in all_results)
        successful_runs = sum(
            sum(1 for result in r["results"] if result["success"])
            for r in all_results
        )

        logger.info("📊 Collection Summary:")
        logger.info(f"  Total test cases: {total_tests}")
        logger.info(f"  Total runs: {total_runs}")
        logger.info(f"  Successful runs: {successful_runs}/{total_runs}")
        logger.info(f"  Success rate: {100 * successful_runs / total_runs:.1f}%")


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect ClinicalBrain data using GPT-4o and Claude 4 models"
    )
    parser.add_argument(
        "--test-cases",
        nargs="+",
        help="Specific test case IDs to run (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        default="clinical_brain_data",
        help="Output directory for collected data (default: clinical_brain_data)"
    )

    args = parser.parse_args()

    print("🧠 ClinicalBrain Data Collection Script")
    print("=" * 80)
    print("This script runs ClinicalBrain locally with multiple model combinations:")
    print("  1. GPT-4o (evaluator) + Claude 4 (responder)")
    print("  2. Claude 4 (evaluator) + GPT-4o (responder)")
    print("=" * 80)
    print()

    # Check environment variables
    required_env_vars = ["AZURE_API_KEY", "AZURE_API_BASE", "ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these in your .env file or environment")
        return

    collector = DataCollector(output_dir=args.output_dir)
    await collector.collect_data(test_case_ids=args.test_cases)


if __name__ == "__main__":
    asyncio.run(main())

def _classify_response_category(response_text: str) -> str:
    # Placeholder - implement based on chosen strategy
    # For now, default to N-GI (most common category)
    return "N-GI"
