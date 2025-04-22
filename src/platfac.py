import sys
import platform
from enum import IntFlag, auto
from abc import ABC, abstractmethod


class ProcessorFeatures(IntFlag):
    """Extensible processor feature detection."""
    BASIC = auto()
    SSE = auto()
    AVX = auto()
    AVX2 = auto()
    AVX512 = auto()
    NEON = auto()
    SVE = auto()
    RVV = auto()  # RISC-V Vector Extensions
    AMX = auto()  # Advanced Matrix Extensions

    @classmethod
    def detect_features(cls) -> 'ProcessorFeatures':
        platform_factory = PlatformFactory()
        detector = platform_factory.get_detector()
        return detector.detect_features()


class PlatformDetector(ABC):
    @abstractmethod
    def detect_features(self) -> ProcessorFeatures:
        pass


class X86Detector(PlatformDetector):
    def detect_features(self) -> ProcessorFeatures:
        features = ProcessorFeatures.BASIC
        try:
            if sys.platform == 'win32':
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                     r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
                identifier = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
            else:
                with open('/proc/cpuinfo') as f:
                    identifier = next(line.split(':')[1] for line in f if 'model name' in line)

            if 'avx512' in identifier.lower():
                features |= ProcessorFeatures.AVX512
            if 'avx2' in identifier.lower():
                features |= ProcessorFeatures.AVX2
            if 'avx' in identifier.lower():
                features |= ProcessorFeatures.AVX
            if 'sse' in identifier.lower():
                features |= ProcessorFeatures.SSE
            if 'fp16' in identifier.lower():
                features |= ProcessorFeatures.AMX
            if 'riscv' in identifier.lower():
                features |= ProcessorFeatures.RVV
            if 'amx' in identifier.lower():
                features |= ProcessorFeatures.AMX
        except Exception:
            pass  # Fallback to basic features
        return features


class ArmDetector(PlatformDetector):
    def detect_features(self) -> ProcessorFeatures:
        features = ProcessorFeatures.BASIC
        try:
            if sys.platform == 'darwin':  # Apple Silicon
                features |= ProcessorFeatures.NEON
            else:
                with open('/proc/cpuinfo') as f:
                    cpuinfo = f.read().lower()
                    if 'neon' in cpuinfo:
                        features |= ProcessorFeatures.NEON
                    if 'sve' in cpuinfo:
                        features |= ProcessorFeatures.SVE
        except Exception:
            pass  # Fallback to basic features
        return features


class PlatformFactory:
    @staticmethod
    def get_detector() -> PlatformDetector:
        machine = platform.machine().lower()
        if machine in ('x86_64', 'amd64', 'x86', 'i386'):
            return X86Detector()
        elif machine.startswith('arm'):
            return ArmDetector()
        else:
            return BasicDetector()  # Fallback to a basic detector


class BasicDetector(PlatformDetector):
    def detect_features(self) -> ProcessorFeatures:
        return ProcessorFeatures.BASIC


# Example usage
if __name__ == "__main__":
    features = ProcessorFeatures.detect_features()
    print(f"Detected features: {features.__dict__}")