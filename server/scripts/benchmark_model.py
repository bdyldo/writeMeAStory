#!/usr/bin/env python3
"""
ML Model Benchmarking Script for CI/CD Pipeline
Tests Modal integration performance and generates metrics
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Any
from pathlib import Path
import sys
import os

# Add server to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.core.modal_story_generator import ModalStoryGenerator


class ModelBenchmark:
    """Comprehensive model performance benchmarking"""
    
    def __init__(self):
        self.generator = ModalStoryGenerator()
        self.results = {
            "timestamp": time.time(),
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "performance_metrics": {},
            "error_analysis": {},
            "test_details": []
        }
    
    async def run_single_benchmark(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Run a single benchmark test"""
        start_time = time.time()
        
        try:
            words = []
            generation_start = time.time()
            
            async for word in self.generator.generate_story_stream(prompt, max_tokens, temperature):
                words.append(word)
            
            generation_end = time.time()
            
            result = {
                "success": True,
                "prompt_length": len(prompt.split()),
                "generated_length": len(''.join(words).split()),
                "generation_time": generation_end - generation_start,
                "total_time": time.time() - start_time,
                "generated_text": ''.join(words)[:100] + "...",  # First 100 chars for logging
                "max_tokens": max_tokens,
                "temperature": temperature,
                "error": None
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "prompt_length": len(prompt.split()),
                "generated_length": 0,
                "generation_time": 0,
                "total_time": time.time() - start_time,
                "generated_text": "",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "error": str(e)
            }
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance tests"""
        print("🚀 Starting Model Performance Benchmarks...")
        
        # Test cases: different prompt lengths and generation parameters
        test_cases = [
            ("Once upon a time", 20, 0.0),  # Short, deterministic
            ("In a world where magic exists", 30, 0.5),  # Medium, creative
            ("The ancient forest was filled with mysterious creatures and hidden secrets", 50, 0.7),  # Long, very creative
            ("Write a story about", 15, 0.3),  # Short, moderate creativity
            ("The spaceship landed on the alien planet and the crew prepared to explore", 40, 0.6)  # Long, creative
        ]
        
        generation_times = []
        total_times = []
        successful_generations = 0
        
        for i, (prompt, max_tokens, temperature) in enumerate(test_cases):
            print(f"📊 Running test {i+1}/{len(test_cases)}: '{prompt[:30]}...'")
            
            result = await self.run_single_benchmark(prompt, max_tokens, temperature)
            
            self.results["test_details"].append(result)
            self.results["total_tests"] += 1
            
            if result["success"]:
                successful_generations += 1
                self.results["successful_tests"] += 1
                generation_times.append(result["generation_time"])
                total_times.append(result["total_time"])
                print(f"✅ Success: {result['generated_length']} words in {result['generation_time']:.2f}s")
            else:
                self.results["failed_tests"] += 1
                print(f"❌ Failed: {result['error']}")
        
        # Calculate performance metrics
        if generation_times:
            self.results["performance_metrics"] = {
                "avg_generation_time": statistics.mean(generation_times),
                "max_generation_time": max(generation_times),
                "min_generation_time": min(generation_times),
                "median_generation_time": statistics.median(generation_times),
                "avg_total_time": statistics.mean(total_times),
                "success_rate": successful_generations / len(test_cases),
                "total_tests_run": len(test_cases)
            }
        
        return self.results
    
    async def run_stress_test(self, concurrent_requests: int = 3) -> Dict[str, Any]:
        """Run stress test with concurrent requests"""
        print(f"🔥 Running stress test with {concurrent_requests} concurrent requests...")
        
        stress_results = {
            "concurrent_requests": concurrent_requests,
            "all_successful": True,
            "individual_results": [],
            "avg_time_under_load": 0
        }
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent_requests):
            prompt = f"Stress test prompt number {i+1}"
            task = self.run_single_benchmark(prompt, 25, 0.5)
            tasks.append(task)
        
        # Run all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_stress_time = time.time() - start_time
        
        successful_stress_tests = 0
        stress_times = []
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result.get("success"):
                successful_stress_tests += 1
                stress_times.append(result["generation_time"])
                stress_results["individual_results"].append(result)
            else:
                stress_results["all_successful"] = False
                print(f"❌ Stress test {i+1} failed: {result}")
        
        if stress_times:
            stress_results["avg_time_under_load"] = statistics.mean(stress_times)
        
        stress_results["total_time"] = total_stress_time
        stress_results["success_rate"] = successful_stress_tests / concurrent_requests
        
        print(f"📈 Stress test completed: {successful_stress_tests}/{concurrent_requests} successful")
        
        return stress_results
    
    def validate_performance_thresholds(self) -> Dict[str, Any]:
        """Validate performance against defined thresholds"""
        thresholds = {
            "max_avg_generation_time": 10.0,  # seconds
            "min_success_rate": 0.8,  # 80%
            "max_individual_generation_time": 15.0  # seconds
        }
        
        validation_results = {
            "passed": True,
            "threshold_checks": [],
            "recommendations": []
        }
        
        metrics = self.results["performance_metrics"]
        
        # Check average generation time
        avg_time_check = {
            "metric": "avg_generation_time",
            "value": metrics.get("avg_generation_time", 0),
            "threshold": thresholds["max_avg_generation_time"],
            "passed": metrics.get("avg_generation_time", 0) <= thresholds["max_avg_generation_time"]
        }
        validation_results["threshold_checks"].append(avg_time_check)
        
        # Check success rate
        success_rate_check = {
            "metric": "success_rate",
            "value": metrics.get("success_rate", 0),
            "threshold": thresholds["min_success_rate"],
            "passed": metrics.get("success_rate", 0) >= thresholds["min_success_rate"]
        }
        validation_results["threshold_checks"].append(success_rate_check)
        
        # Check max individual time
        max_time_check = {
            "metric": "max_generation_time",
            "value": metrics.get("max_generation_time", 0),
            "threshold": thresholds["max_individual_generation_time"],
            "passed": metrics.get("max_generation_time", 0) <= thresholds["max_individual_generation_time"]
        }
        validation_results["threshold_checks"].append(max_time_check)
        
        # Determine overall pass/fail
        validation_results["passed"] = all(check["passed"] for check in validation_results["threshold_checks"])
        
        # Generate recommendations
        if not avg_time_check["passed"]:
            validation_results["recommendations"].append("Consider optimizing model inference time")
        
        if not success_rate_check["passed"]:
            validation_results["recommendations"].append("Investigate and fix reliability issues")
        
        if not max_time_check["passed"]:
            validation_results["recommendations"].append("Add request timeout handling for slow requests")
        
        return validation_results
    
    def generate_report(self, stress_results: Dict[str, Any], validation_results: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report"""
        report_lines = [
            "=" * 60,
            "🤖 ML MODEL BENCHMARK REPORT",
            "=" * 60,
            "",
            "📊 PERFORMANCE SUMMARY:",
            f"  • Total tests: {self.results['total_tests']}",
            f"  • Successful: {self.results['successful_tests']}",
            f"  • Failed: {self.results['failed_tests']}",
            f"  • Success rate: {self.results['performance_metrics'].get('success_rate', 0):.1%}",
            "",
            "⏱️  TIMING METRICS:",
            f"  • Average generation time: {self.results['performance_metrics'].get('avg_generation_time', 0):.2f}s",
            f"  • Median generation time: {self.results['performance_metrics'].get('median_generation_time', 0):.2f}s",
            f"  • Min generation time: {self.results['performance_metrics'].get('min_generation_time', 0):.2f}s",
            f"  • Max generation time: {self.results['performance_metrics'].get('max_generation_time', 0):.2f}s",
            "",
            "🔥 STRESS TEST RESULTS:",
            f"  • Concurrent requests: {stress_results['concurrent_requests']}",
            f"  • Success rate under load: {stress_results['success_rate']:.1%}",
            f"  • Average time under load: {stress_results['avg_time_under_load']:.2f}s",
            f"  • Total stress test time: {stress_results['total_time']:.2f}s",
            "",
            "✅ VALIDATION RESULTS:",
            f"  • Overall validation: {'PASSED' if validation_results['passed'] else 'FAILED'}",
        ]
        
        for check in validation_results["threshold_checks"]:
            status = "✅ PASS" if check["passed"] else "❌ FAIL"
            report_lines.append(f"  • {check['metric']}: {check['value']:.2f} (threshold: {check['threshold']}) {status}")
        
        if validation_results["recommendations"]:
            report_lines.extend([
                "",
                "💡 RECOMMENDATIONS:",
            ])
            for rec in validation_results["recommendations"]:
                report_lines.append(f"  • {rec}")
        
        report_lines.extend([
            "",
            "=" * 60,
            ""
        ])
        
        return "\n".join(report_lines)
    
    async def run_full_benchmark(self) -> bool:
        """Run complete benchmark suite and return success status"""
        try:
            # Run performance tests
            await self.run_performance_tests()
            
            # Run stress test
            stress_results = await self.run_stress_test()
            
            # Validate performance
            validation_results = self.validate_performance_thresholds()
            
            # Generate and print report
            report = self.generate_report(stress_results, validation_results)
            print(report)
            
            # Save detailed results to file
            results_file = Path("benchmark_results.json")
            with open(results_file, "w") as f:
                full_results = {
                    "performance_tests": self.results,
                    "stress_tests": stress_results,
                    "validation": validation_results
                }
                json.dump(full_results, f, indent=2)
            
            print(f"📄 Detailed results saved to: {results_file}")
            
            return validation_results["passed"]
            
        except Exception as e:
            print(f"❌ Benchmark failed with error: {e}")
            return False


async def main():
    """Main benchmark execution"""
    print("🤖 Starting ML Model Benchmarking Suite...")
    
    # Check if running in CI environment
    is_ci = os.getenv("CI", "false").lower() == "true"
    if is_ci:
        print("🔧 Running in CI environment")
    
    benchmark = ModelBenchmark()
    success = await benchmark.run_full_benchmark()
    
    if success:
        print("🎉 All benchmarks passed!")
        sys.exit(0)
    else:
        print("❌ Benchmarks failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())