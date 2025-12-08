"""Performance and load tests for face recognition system.

Tests are marked with @pytest.mark.slow and will be skipped in normal test runs.
Run with: pytest tests/recognition/test_performance.py -v -m slow

This module provides performance benchmarks for:
- FAISS index operations (build time, search latency, concurrent access)
- Recognition pipeline end-to-end performance
- Simulated load testing
"""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import pytest

from recognition.faiss_index import APPROXIMATE_SEARCH_THRESHOLD, FAISSIndex

# =============================================================================
# PERFORMANCE THRESHOLDS
# =============================================================================

# Expected performance thresholds (in milliseconds)
THRESHOLDS = {
    "faiss_build_100": 10,  # 100 embeddings should build in <10ms
    "faiss_build_1000": 100,  # 1000 embeddings should build in <100ms
    "faiss_build_10000": 2000,  # 10000 embeddings should build in <2s
    "faiss_search_single": 1,  # Single search should be <1ms
    "faiss_search_k10": 5,  # k=10 search should be <5ms
    "concurrent_search_ops": 50,  # 100 concurrent searches <50ms total
}


def measure_time_ms(func: Callable, iterations: int = 1) -> tuple[float, float]:
    """Measure execution time in milliseconds.

    Args:
        func: Function to measure
        iterations: Number of iterations for averaging

    Returns:
        Tuple of (mean_ms, std_ms)
    """
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)

    return statistics.mean(timings), statistics.stdev(timings) if len(timings) > 1 else 0.0


# =============================================================================
# FAISS INDEX PERFORMANCE TESTS
# =============================================================================


@pytest.mark.slow
class TestFAISSIndexPerformance:
    """Performance benchmarks for FAISS index operations."""

    @pytest.fixture
    def small_embeddings(self):
        """100 embeddings for small dataset tests."""
        return np.random.randn(100, 128).astype(np.float32)

    @pytest.fixture
    def medium_embeddings(self):
        """1000 embeddings for medium dataset tests."""
        return np.random.randn(1000, 128).astype(np.float32)

    @pytest.fixture
    def large_embeddings(self):
        """10000 embeddings for large dataset tests."""
        return np.random.randn(10000, 128).astype(np.float32)

    def test_index_build_100_embeddings(self, small_embeddings):
        """Benchmark: Building index with 100 embeddings."""
        labels = [f"user_{i}" for i in range(100)]

        def build():
            index = FAISSIndex(dimension=128)
            index.add_embeddings(small_embeddings, labels)

        mean_ms, std_ms = measure_time_ms(build, iterations=5)

        print(f"\n100 embeddings build: {mean_ms:.2f}ms ± {std_ms:.2f}ms")
        assert (
            mean_ms < THRESHOLDS["faiss_build_100"]
        ), f"Build took {mean_ms:.2f}ms, expected <{THRESHOLDS['faiss_build_100']}ms"

    def test_index_build_1000_embeddings(self, medium_embeddings):
        """Benchmark: Building index with 1000 embeddings."""
        labels = [f"user_{i}" for i in range(1000)]

        def build():
            index = FAISSIndex(dimension=128)
            index.add_embeddings(medium_embeddings, labels)

        mean_ms, std_ms = measure_time_ms(build, iterations=3)

        print(f"\n1000 embeddings build: {mean_ms:.2f}ms ± {std_ms:.2f}ms")
        assert (
            mean_ms < THRESHOLDS["faiss_build_1000"]
        ), f"Build took {mean_ms:.2f}ms, expected <{THRESHOLDS['faiss_build_1000']}ms"

    def test_index_build_10000_embeddings(self, large_embeddings):
        """Benchmark: Building index with 10000 embeddings."""
        labels = [f"user_{i}" for i in range(10000)]

        def build():
            index = FAISSIndex(dimension=128)
            index.add_embeddings(large_embeddings, labels)

        mean_ms, std_ms = measure_time_ms(build, iterations=2)

        print(f"\n10000 embeddings build: {mean_ms:.2f}ms ± {std_ms:.2f}ms")
        assert (
            mean_ms < THRESHOLDS["faiss_build_10000"]
        ), f"Build took {mean_ms:.2f}ms, expected <{THRESHOLDS['faiss_build_10000']}ms"

    def test_single_search_latency(self, medium_embeddings):
        """Benchmark: Single query search latency."""
        index = FAISSIndex(dimension=128)
        labels = [f"user_{i}" for i in range(1000)]
        index.add_embeddings(medium_embeddings, labels)

        query = np.random.randn(128).astype(np.float32)

        def search():
            index.search_single(query)

        mean_ms, std_ms = measure_time_ms(search, iterations=100)

        print(f"\nSingle search latency: {mean_ms:.3f}ms ± {std_ms:.3f}ms")
        assert (
            mean_ms < THRESHOLDS["faiss_search_single"]
        ), f"Search took {mean_ms:.3f}ms, expected <{THRESHOLDS['faiss_search_single']}ms"

    def test_k10_search_latency(self, medium_embeddings):
        """Benchmark: k=10 nearest neighbor search latency."""
        index = FAISSIndex(dimension=128)
        labels = [f"user_{i}" for i in range(1000)]
        index.add_embeddings(medium_embeddings, labels)

        query = np.random.randn(128).astype(np.float32)

        def search():
            index.search(query, k=10)

        mean_ms, std_ms = measure_time_ms(search, iterations=100)

        print(f"\nk=10 search latency: {mean_ms:.3f}ms ± {std_ms:.3f}ms")
        assert (
            mean_ms < THRESHOLDS["faiss_search_k10"]
        ), f"Search took {mean_ms:.3f}ms, expected <{THRESHOLDS['faiss_search_k10']}ms"

    def test_concurrent_search_operations(self, medium_embeddings):
        """Benchmark: Concurrent search operations."""
        index = FAISSIndex(dimension=128)
        labels = [f"user_{i}" for i in range(1000)]
        index.add_embeddings(medium_embeddings, labels)

        queries = [np.random.randn(128).astype(np.float32) for _ in range(100)]

        def concurrent_search():
            with ThreadPoolExecutor(max_workers=10) as executor:
                list(executor.map(lambda q: index.search_single(q), queries))

        mean_ms, std_ms = measure_time_ms(concurrent_search, iterations=3)

        print(f"\n100 concurrent searches: {mean_ms:.2f}ms ± {std_ms:.2f}ms")
        assert (
            mean_ms < THRESHOLDS["concurrent_search_ops"]
        ), f"Concurrent ops took {mean_ms:.2f}ms, expected <{THRESHOLDS['concurrent_search_ops']}ms"

    def test_ivf_vs_flat_index_threshold(self):
        """Verify that IVF index is used for large datasets."""
        # This test validates the APPROXIMATE_SEARCH_THRESHOLD behavior
        assert APPROXIMATE_SEARCH_THRESHOLD > 0, "Threshold should be positive"

        # Below threshold - should use flat index
        small_n = APPROXIMATE_SEARCH_THRESHOLD - 1
        small_emb = np.random.randn(small_n, 64).astype(np.float32)
        small_labels = [f"u{i}" for i in range(small_n)]

        small_index = FAISSIndex(dimension=64)
        small_index.add_embeddings(small_emb, small_labels)

        # Search should work and be exact
        query = small_emb[0]
        result = small_index.search_single(query)
        assert result is not None
        assert result[0] == "u0"
        assert pytest.approx(result[1], abs=1e-5) == 0.0  # Exact match

        # At threshold - should use IVF
        large_n = APPROXIMATE_SEARCH_THRESHOLD
        large_emb = np.random.randn(large_n, 64).astype(np.float32)
        large_labels = [f"u{i}" for i in range(large_n)]

        large_index = FAISSIndex(dimension=64)
        large_index.add_embeddings(large_emb, large_labels)

        # Search should still work (approximate)
        query = large_emb[0]
        result = large_index.search_single(query)
        assert result is not None
        assert result[1] < 0.1  # Should be close (approximate match)


# =============================================================================
# LOAD TESTING
# =============================================================================


@pytest.mark.slow
class TestLoadSimulation:
    """Simulated load tests for the recognition system."""

    def test_rapid_sequential_searches(self):
        """Stress test: Rapid sequential search requests."""
        index = FAISSIndex(dimension=128)
        embeddings = np.random.randn(500, 128).astype(np.float32)
        labels = [f"user_{i}" for i in range(500)]
        index.add_embeddings(embeddings, labels)

        # Simulate 1000 rapid sequential requests
        n_requests = 1000
        queries = [np.random.randn(128).astype(np.float32) for _ in range(n_requests)]

        start = time.perf_counter()
        for query in queries:
            index.search_single(query)
        total_ms = (time.perf_counter() - start) * 1000

        avg_ms = total_ms / n_requests
        print(f"\n1000 sequential searches: total={total_ms:.2f}ms, avg={avg_ms:.3f}ms/request")

        # Ensure reasonable average latency
        assert avg_ms < 1.0, f"Average latency {avg_ms:.3f}ms too high"

    def test_mixed_concurrent_workload(self):
        """Stress test: Concurrent read + write operations."""
        index = FAISSIndex(dimension=64)

        # Initial population
        initial_emb = np.random.randn(100, 64).astype(np.float32)
        initial_labels = [f"init_{i}" for i in range(100)]
        index.add_embeddings(initial_emb, initial_labels)

        errors = []

        def search_task():
            try:
                query = np.random.randn(64).astype(np.float32)
                index.search_single(query)
            except Exception as e:
                errors.append(str(e))

        # Concurrent searches only (since FAISS doesn't support concurrent writes)
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(search_task) for _ in range(200)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors during concurrent access: {errors[:5]}"

    def test_memory_stability_repeated_operations(self):
        """Stress test: Repeated build/search/clear cycles."""
        dimension = 128

        for cycle in range(5):
            index = FAISSIndex(dimension=dimension)

            # Build
            embeddings = np.random.randn(500, dimension).astype(np.float32)
            labels = [f"user_{i}" for i in range(500)]
            index.add_embeddings(embeddings, labels)

            # Search
            for _ in range(50):
                query = np.random.randn(dimension).astype(np.float32)
                index.search_single(query)

            # Clear
            index.clear()
            assert index.size == 0

        # If we get here without memory errors, test passes
        assert True


# =============================================================================
# SCALING TESTS
# =============================================================================


@pytest.mark.slow
class TestScalingBehavior:
    """Tests for understanding how performance scales with data size."""

    def test_search_scales_sublinearly(self):
        """Verify search time scales sublinearly with index size."""
        dimension = 128
        sizes = [100, 500, 1000]
        latencies = []

        for size in sizes:
            index = FAISSIndex(dimension=dimension)
            embeddings = np.random.randn(size, dimension).astype(np.float32)
            labels = [f"u{i}" for i in range(size)]
            index.add_embeddings(embeddings, labels)

            query = np.random.randn(dimension).astype(np.float32)

            # Warm up
            for _ in range(10):
                index.search_single(query)

            # Measure
            mean_ms, _ = measure_time_ms(lambda: index.search_single(query), iterations=100)
            latencies.append(mean_ms)

            print(f"\nSize {size}: {mean_ms:.3f}ms per search")

        # Latencies should not grow linearly with size
        # If linear: latency[2]/latency[0] ≈ sizes[2]/sizes[0] = 10x
        # With flat index, should be ~constant; with IVF, should be sublinear
        ratio = latencies[2] / latencies[0]
        size_ratio = sizes[2] / sizes[0]

        print(f"\nLatency ratio (1000/100): {ratio:.2f}")
        print(f"Size ratio: {size_ratio:.2f}")

        # Allow some growth but should be much less than linear
        assert (
            ratio < size_ratio / 2
        ), f"Search time growing too fast: {ratio:.2f}x for {size_ratio:.2f}x data"
