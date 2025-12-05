"""Management command to profile hardware and model performance."""

from django.core.management.base import BaseCommand

from recognition.performance_utils import (
    detect_hardware,
    get_recommended_config,
    profile_model_performance,
)


class Command(BaseCommand):
    """Profile available hardware accelerators and model performance."""

    help = "Detect available hardware (NPU/GPU/CPU) and profile model performance"

    def add_arguments(self, parser):
        parser.add_argument(
            "--models",
            nargs="+",
            default=["Facenet", "VGG-Face", "OpenFace"],
            help="Models to profile (default: Facenet VGG-Face OpenFace)",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=5,
            help="Number of iterations for profiling (default: 5)",
        )
        parser.add_argument(
            "--skip-profiling",
            action="store_true",
            help="Skip model profiling, only detect hardware",
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("=== Hardware Detection ===\n"))

        # Detect hardware
        hardware = detect_hardware()
        hardware_dict = hardware.to_dict()

        # Display NPU info
        if hardware_dict["npu"]["available"]:
            self.stdout.write(
                self.style.SUCCESS(
                    f"✓ NPU Detected: {hardware_dict['npu']['type']} "
                    f"({hardware_dict['npu']['device']}) "
                    f"via {hardware_dict['npu']['backend']}"
                )
            )
        else:
            self.stdout.write(self.style.WARNING(" NPU: Not available"))

        # Display GPU info
        if hardware_dict["gpu"]["available"]:
            gpu_name = hardware_dict["gpu"]["name"] or "Unknown GPU"
            memory_info = ""
            if hardware_dict["gpu"]["memory_mb"]:
                memory_info = f" ({hardware_dict['gpu']['memory_mb']} MB)"
            self.stdout.write(self.style.SUCCESS(f"✓ GPU Detected: {gpu_name}{memory_info}"))
        else:
            self.stdout.write(self.style.WARNING("✗ GPU: Not available"))

        # Display CPU info
        self.stdout.write(self.style.SUCCESS("✓ CPU: Available"))

        # Get recommended configuration
        self.stdout.write(self.style.SUCCESS("\n=== Recommended Configuration ===\n"))
        config = get_recommended_config(hardware)

        self.stdout.write(f"Backend: {self.style.SUCCESS(config['backend'])}")
        self.stdout.write(f"Model: {self.style.SUCCESS(config['model'])}")
        self.stdout.write(f"Detector: {config['detector_backend']}")

        if config["backend"] == "openvino":
            self.stdout.write(f"OpenVINO Device: {config.get('openvino_device', 'N/A')}")

        # Profile models if not skipped
        if not options["skip_profiling"]:
            self.stdout.write(self.style.SUCCESS("\n=== Model Performance Profiling ===\n"))
            self.stdout.write(
                f"Profiling {len(options['models'])} models with {options['iterations']} iterations each..."
            )
            self.stdout.write("(This may take a few minutes)\n")

            results = {}
            for model_name in options["models"]:
                self.stdout.write(f"Profiling {model_name}...", ending=" ")
                metrics = profile_model_performance(
                    model_name, num_iterations=options["iterations"]
                )

                if metrics:
                    results[model_name] = metrics
                    self.stdout.write(self.style.SUCCESS(f"✓ {metrics['mean_ms']:.2f}ms avg"))
                else:
                    self.stdout.write(self.style.ERROR("✗ Failed"))

            # Display summary table
            if results:
                self.stdout.write("\n=== Performance Summary ===\n")
                self.stdout.write(
                    f"{'Model':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)'}"
                )
                self.stdout.write("-" * 65)

                for model_name, metrics in sorted(results.items(), key=lambda x: x[1]["mean_ms"]):
                    self.stdout.write(
                        f"{model_name:<15} "
                        f"{metrics['mean_ms']:<12.2f} "
                        f"{metrics['std_ms']:<12.2f} "
                        f"{metrics['min_ms']:<12.2f} "
                        f"{metrics['max_ms']:.2f}"
                    )

                # Fastest model
                fastest = min(results.items(), key=lambda x: x[1]["mean_ms"])
                self.stdout.write(
                    self.style.SUCCESS(f"\nFastest: {fastest[0]} ({fastest[1]['mean_ms']:.2f}ms)")
                )

        self.stdout.write(self.style.SUCCESS("\n✓ Hardware profiling complete!"))
